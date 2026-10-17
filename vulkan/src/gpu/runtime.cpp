// vulkan/src/gpu/runtime.cpp - the Vulkan device runtime (Phase 1 S3).
//
// WHAT S3 HAD TO FIX, AND WHERE THE DEVIATION WAS.
// S2 left this file as the *face* of the runtime with no state behind it: `device_count()` answered "unsupported",
// `malloc_device_raw` returned null, `host_view` answered nullptr, `mem_get_info` had no numbers, and events / streams
// / graphs had no semaphore, no queue and no command buffer.  Nothing was missing at the *call* level - every name
// existed - what was missing was the state those calls operate on.  The chain, in order:
//   no vkCreateInstance / vkEnumeratePhysicalDevices / vkCreateDevice call existed anywhere in the file (ctx() was
//   declared and never defined)  ->  no VkDevice and no VkQueue  ->  nothing to vkAllocateMemory against, nothing to
//   record into, nothing to signal, nothing to wait on  ->  every entry point that needs a device had nothing to do
//   but refuse.  S3 is that state, built in the order the data flows: instance -> physical devices -> the selection
//   predicate -> VkDevice -> queues -> VkDeviceMemory blocks -> sub-allocations -> command buffers -> submits ->
//   timeline values -> captured graphs -> the CUDA-facing face.
//
// THE TWO PHASE 0 MEASUREMENTS THIS FILE IS WRITTEN AGAINST (bench/results/vulkan-caps-2026-10-17.md):
//   * P5: a `coherent volatile` flag in memory the host can write, polled by a shader, was never observed by that
//     shader while its own dispatch ran - in all four host-visible flavours and with both host write methods.  So
//     nothing in this port makes the GPU notice something by reading memory the host wrote: a host->GPU signal is a
//     SUBMIT (nudge_submit), a GPU->host signal is a timeline semaphore value (wait_timeline).  No shader under
//     vulkan/shaders/ has a spin or a poll loop (the grep is recorded in vulkan/PHASE1-S3-RUNTIME.md), and the
//     probe's spin shader lives in vulkan/probe/shaders/ as the instrument that measured P5, never as something this
//     runtime dispatches.
//   * P1/P2/P2b: both Arcs report maxStorageBufferRange 4095 MiB, maxBufferSize (maintenance4) 4096 MiB and
//     maxMemoryAllocationSize (maintenance3) 4095 MiB, VK_EXT_shader_64bit_indexing is absent so no feature lifts them,
//     minStorageBufferOffsetAlignment is 64, a *binding* may not cover more than 4095 MiB while a descriptor *offset*
//     may sit anywhere including past 4 GiB (P2b: offset 4096 MiB with a 512 MiB range read and wrote correct tags on
//     both cards), and a device address reaches 5056 MiB inside one allocation.  Those numbers are read from the card
//     at run time (read_limits, block_limit), not hardcoded; the clamp and the offset rule are in the code below.
//
// THE MAP, IN CODE (the same table VULKAN-BACKEND-PLAN.md's mapping section gives):
//   device   one VkDevice per discrete Intel Arc, kept by a property test on each enumerated physical device
//            (choose_physical); the integrated UHD 770 is dropped by that test, never by dropping an array index.
//   memory   VkDeviceMemory blocks no larger than min(maxMemoryAllocationSize, maxBufferSize, maxStorageBufferRange),
//            one VkBuffer over the whole block, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT on EVERY allocation, spans
//            sub-allocated out of the blocks with every offset rounded up so the device pointer honours the card's
//            minStorageBufferOffsetAlignment.
//   streams  one VkQueue per stream (as many as the card hands out; streams beyond that share a queue as a serialized
//            submit chain), every submit carrying a fence and a signal on that stream's timeline semaphore.
//   events   timeline semaphores: a record is a signal entry in the submit that carries the stream's recorded work (or
//            vkSignalSemaphore when the stream has recorded nothing), a wait is a wait entry in the next submit (or
//            vkWaitSemaphores on the host).
//   graphs   capture records ONCE into one VkCommandBuffer; cudaGraphLaunch vkResetCommandBuffers that same buffer,
//            re-records the node list into it and vkQueueSubmits it again.  Per-window values stay in the engine's own
//            device buffers, so a replay needs no graph update.
//   host     HOST_VISIBLE|HOST_COHERENT blocks (mapped once for the block's life) for staging, parameter blocks and
//            window values, plus VK_EXT_external_memory_host imports of host memory the engine already owns - which is
//            what cudaHostAlloc(Mapped|Portable), cudaHostRegister and cudaHostGetDevicePointer are answered with
//            (P3: one host allocation imported into both cards reads correct data in place).
//   queries  cudaMemGetInfo from VK_EXT_memory_budget's per-heap budget minus this process's own usage in that heap.
//   launches launch_kernel dispatches a registered SPIR-V pipeline; register_pipeline is where a Phase 2 kernel plugs
//            in.  An entry point that is not registered is named and refused, never faked.
//
// STRATA_VULKAN_SELFCHECK (selfcheck(), in the context section) prints once per process - and only for a run that
// actually opens the runtime, because --help / --version / --families return before any strata::gpu:: call - the
// devices the selection kept and the ones it dropped with the reason each got, the limits, the heaps with their budget
// and usage, and the import alignment.  STRATA_VULKAN_SELFCHECK=0 silences it.
// The environment switches this runtime reads (STRATA_VULKAN_SELFCHECK, STRATA_VULKAN_IGNORE_ALLOC_LIMIT,
// STRATA_VULKAN_GRAPH_REUPLOAD) are read with getenv, which MSVC deprecates by name rather than by use; this is the
// line that says so before any system header is pulled in, instead of /D on the target where the next file would hit
// the same C4996.
#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "strata/gpu/runtime.hpp"

#include <cuda.h>          // vulkan_compat/cuda.h: the cuX signatures driver_entry_point is asked for by name.
                           // src/core/vmm.cpp:16-22 takes them as `decltype(&cuMemMap)` and resolves them by NAME
                           // (vmm.cpp:25-39), so nothing in ../src links against them; this port answers them
                           // "not found" until the segmented arena exists, and vmm.cpp:33-36 already treats that
                           // answer as "the VMM path is off" rather than as a failure.
#include <cuda_runtime.h>  // vulkan_compat/cuda_runtime.h: cudaGraphNodeType... is what NodeImpl::type holds
#include <vulkan/vulkan.h> // the SDK headers vulkan/CMakeLists.txt finds (find_package(Vulkan))

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#  include <malloc.h>      // _aligned_malloc / _aligned_free: the 4 KiB-aligned host allocations an import can take
#else
#  include <stdlib.h>
#endif

#if defined(__has_include)
#  if __has_include("vulkan_shaders/noop.h")
#    include "vulkan_shaders/noop.h"          // generated from vulkan/shaders/noop.comp (vulkan/CMakeLists.txt)
#    define STRATA_VULKAN_HAS_NOOP_SPV 1
#  endif
#endif

namespace strata::gpu {
// ------------------------------------------------------------------ constants and small helpers

constexpr uint64_t kKiB = 1024ull;
constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint64_t kGiB = 1024ull * 1024ull * 1024ull;

// The floor for a block when the engine asks for something small.  A block is never bigger than the card's own limit
// (block_limit() below); these two only stop a 4 KiB decode allocation from opening a 4 GiB block.
constexpr uint64_t kDeviceBlockBytes = 128ull * kMiB;
constexpr uint64_t kHostBlockBytes   = 16ull * kMiB;

// One VkQueue per stream, up to this many per device; a card that hands out fewer gets fewer, and the streams past
// that share a queue as a serialized submit chain (a VkQueue runs its submits in submission order).
constexpr uint32_t kStreamQueues = 8;
constexpr uint32_t kChainSlots   = 4;      // command buffers / fences in flight per stream (a serialised chain)
constexpr uint32_t kFlushEvery   = 4096;   // recorded commands in one chain slot before an automatic submit
constexpr uint64_t kWaitSliceNs  = 5ull * 1000ull * 1000ull * 1000ull;   // vkWaitSemaphores slice: see wait_timeline
constexpr uint64_t kFenceWaitNs  = 60ull * 1000ull * 1000ull * 1000ull;  // a fence wait is bounded, not infinite
constexpr uint64_t kParamBytes   = 4096;   // one chain slot's uniform parameter block

// Host-mapped memory in this port is host-visible AND host-coherent, always: a host write followed by a submit is
// what the engine's cudaHostAlloc(Mapped) buffers rely on, and vkFlushMappedMemoryRanges is the workaround this
// avoids.  Phase 0 P1 read the types both Arcs offer (bench/results/vulkan-caps-gate-run-1.log:20-21, 57-58):
//   type  1 heap 1: HOST_VISIBLE HOST_COHERENT
//   type  2 heap 1: HOST_VISIBLE HOST_COHERENT HOST_CACHED
constexpr VkMemoryPropertyFlags kHostWant = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
// Device blocks ask for the device-local type that is ALSO host-visible and host-coherent, because both Arcs have one
// (bench/results/vulkan-caps-gate-run-1.log:22 on the A770 "type  3 heap 0: DEV_LOCAL HOST_VISIBLE HOST_COHERENT",
// :59 on the B580 "type  4 heap 0: DEV_LOCAL HOST_VISIBLE HOST_COHERENT") and because
// vulkan/src/core/device_vk.cpp:131 writes DeviceArena's poison through strata::gpu::host_view.  sub_alloc falls back
// to plain DEVICE_LOCAL on a card that cannot do both.
constexpr VkMemoryPropertyFlags kDeviceWant = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags kDeviceWantStrict = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
// ------------------------------------------------------------------ the runtime's objects

struct Device;
struct StreamImpl;
struct EventImpl;

// One physical chunk: a VkDeviceMemory with ONE VkBuffer over the whole of it.  That buffer is what the device
// addresses the engine's pointers are made of, what a descriptor binding points at, and what vkCmdCopyBuffer and
// vkCmdFillBuffer write into.  A host-visible block is mapped once, for the block's life: that mapping is what
// strata::gpu::host_view answers with, and what DeviceArena's poison path writes through
// (vulkan/src/core/device_vk.cpp:131-135).
struct Hole { uint64_t off = 0, bytes = 0; };

struct Block {
    Device* dev = nullptr;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceAddress addr = 0;          // vkGetBufferDeviceAddress(buf): the base the pointers in this block are made of
    VkDeviceSize size = 0, used = 0;   // used = the sub-allocator's high-water mark
    uint64_t live = 0;                 // bytes currently handed out of this block
    uint32_t mem_type = 0;
    VkMemoryPropertyFlags props = 0;
    void* map = nullptr;               // HOST_VISIBLE: mapped once (nullptr for a device-only memory type)
    bool imported = false;             // VK_EXT_external_memory_host: the pages belong to a host allocation
    void* host_ptr = nullptr;          // imported: that allocation, rounded DOWN to the import alignment
    std::vector<Hole> holes;           // freed sub-ranges, kept sorted, merged when they touch
};

// A sub-range of a block.  An allocation bigger than the card's per-block limit is several spans whose device
// addresses were checked to be contiguous, because the engine's device pointers ARE arithmetic.
struct Span { Block* block = nullptr; uint64_t block_off = 0; uint64_t addr = 0; uint64_t bytes = 0; };

struct Alloc {
    Device* dev = nullptr;
    std::vector<Span> spans;
    uint64_t addr = 0;                 // a device allocation: the device pointer handed to the engine
    void* host = nullptr;              // a host allocation: the CPU address handed to the engine instead
    uint64_t bytes = 0;
    bool host_alloc = false;           // cudaHostAlloc / cudaMallocHost / cudaHostRegister
    bool imported = false;             // the pages are a host allocation this port does not own
    uint64_t align = 1;
};

struct Pipeline {
    std::string key, name, entry;      // launch name / the SPIR-V entry point (GLSL compiles to "main")
    VkShaderModule mod = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    uint32_t local[3] = {1, 1, 1};
    uint32_t shared = 0;
    uint32_t param_bytes = 0;
    uint32_t storage_count = 0;        // uint64 device addresses in the parameter block that become storage bindings
    uint32_t param_binding = 0;
    bool push_params = true;           // false: the parameter block goes in as a uniform binding instead
};

// A recorded node of a captured graph.  The graph's VkCommandBuffer holds the real vkCmd* stream recorded ONCE at
// capture; the node list is what cudaGraphGetNodes / cudaGraphNodeGetType / cudaGraphKernelNodeGetParams /
// cudaGraphGetEdges report (src/core/verify.cpp:1947-2006), and what cudaGraphLaunch re-encodes into the same buffer.
struct NodeImpl {
    int type = (int) cudaGraphNodeTypeKernel;   // vulkan_compat/cuda_runtime.h:251-254
    std::string name;
    Dim3 grid, block;
    size_t shared = 0;
    std::vector<uint8_t> params;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    int value = 0;
    EventImpl* ev = nullptr;                   // a captured event record / wait: its value is resolved at launch
    HostFn fn = nullptr;
    void* fn_arg = nullptr;
};

struct GraphImpl {
    Device* dev = nullptr;
    VkCommandBuffer cb = VK_NULL_HANDLE;    // recorded ONCE, in this one buffer
    VkCommandPool pool = VK_NULL_HANDLE;    // the pool that buffer came from (it is freed from there)
    Alloc* param_alloc = nullptr;           // the captured dispatches' parameter bytes, in host-visible memory
    uint64_t param_used = 0;
    std::vector<NodeImpl*> nodes;
    int refs = 1;                           // the graph plus every cudaGraphExec_t made from it
    bool ended = false;
};

struct GraphExecImpl {
    Device* dev = nullptr;
    GraphImpl* graph = nullptr;
    VkCommandBuffer cb = VK_NULL_HANDLE;    // the SAME VkCommandBuffer the capture recorded, re-submitted per window
    VkFence fence = VK_NULL_HANDLE;         // the fence its submit carries
    bool inflight = false;
    Alloc* window_alloc = nullptr;          // host-visible host-coherent block holding the per-launch parameter bytes
    uint64_t window_bytes = 0;
    uint64_t window_used = 0;               // the re-record path writes each launch's parameters here
    std::map<std::string, VkDescriptorSet> sets;   // the replay's own descriptor sets (a chain slot has its own)
    bool reencode = false;                  // STRATA_VULKAN_GRAPH_REUPLOAD=1: re-record even on the first launch
    uint64_t launches = 0;
};

struct EventImpl {
    Device* dev = nullptr;
    VkSemaphore sem = VK_NULL_HANDLE;       // a timeline semaphore, created in event_create
    uint64_t value = 0;                     // 0: never recorded, which is CUDA's "a fresh event is complete"
};

// A chain slot: one recorded command buffer, the fence its submit carries, the timeline value that submit signalled,
// and the uniform parameter block the dispatches recorded in it wrote.
struct Chain {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool recording = false;
    bool live = false;                              // submitted and not yet retired
    Alloc* param_alloc = nullptr;                   // the uniform parameter block (push-constant pipelines skip it)
    uint64_t param_used = 0;
    std::map<std::string, VkDescriptorSet> sets;    // one descriptor set per pipeline, per chain slot
};

struct StreamImpl {
    Device* d = nullptr;
    uint32_t qidx = 0;
    VkQueue q = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    Chain chain[kChainSlots];
    uint32_t chain_next = 0;
    Chain* open = nullptr;                          // the chain being recorded, if any
    VkSemaphore tl = VK_NULL_HANDLE;                // this stream's timeline semaphore
    uint64_t signaled = 0;                          // the value its last submit signalled on it
    std::vector<std::pair<VkSemaphore, uint64_t>> waits;    // cudaStreamWaitEvent: waits the next submit carries
    std::vector<std::pair<VkSemaphore, uint64_t>> signals;  // cudaEventRecord: signals the next submit carries
    GraphImpl* rec = nullptr;                       // non-null while this stream is capturing
    uint32_t prio_cuda = 0;                         // the CUDA priority bucket it was created with
    uint32_t recorded = 0;                          // commands recorded in the open chain slot
    uint64_t dispatches = 0, submits = 0, copies = 0, fills = 0;
};

struct Device {
    uint32_t ordinal = 0;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceMaintenance4Properties m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    bool has_m4 = false, has_hostp = false, has_idp = false;
    bool m4_enabled = false;               // VK_KHR_maintenance4 actually enabled on this VkDevice (read_limits only
                                           // chains its struct when it is: a feature or struct for an extension that was
                                           // not enabled is a validation error)
    VkPhysicalDeviceMemoryProperties mem{};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    bool has_budget = false;
    bool has_host_import = false;
    bool has_memory_budget_ext = false;

    uint32_t qfam = 0;
    uint32_t qfam_queues = 0;
    std::vector<VkQueue> queues;
    std::vector<float> prio;

    PFN_vkGetBufferDeviceAddress get_address = nullptr;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_memory = nullptr;

    // The limit every block and every single storage-buffer binding has to respect, and the alignment every
    // sub-allocation offset is rounded up to.  Both are read from the card, never assumed.
    uint64_t block_limit = 0;
    uint64_t align_storage = 64;
    uint64_t align_uniform = 64;
    uint32_t heap_local = 0;
    uint64_t heap_local_bytes = 0;

    std::vector<std::unique_ptr<Block>> blocks;
    std::vector<std::unique_ptr<Alloc>> allocs;
    std::map<uint64_t, Alloc*> by_dev;              // device address -> allocation (interval map, keyed by start)
    std::map<uint64_t, Alloc*> by_host;             // CPU address -> allocation (host allocations and imports)
    std::map<uint64_t, uint64_t> host_imports;      // aligned host base -> that import's Alloc, on this device
    uint64_t own_device_bytes = 0;                  // this process's own live device allocations: cudaMemGetInfo's subtrahend
    Alloc* staging = nullptr;                       // the HOST_VISIBLE|HOST_COHERENT buffer unregistered copies go through
    uint64_t staging_bytes = 0;
    StreamImpl* staging_stream = nullptr;           // the stream whose last submit read or wrote that staging buffer
    uint64_t staging_value = 0;
    bool staging_pending = false;
    // A copy whose destination is host memory this port does not own is recorded into the staging buffer; the bytes
    // are handed to the destination when the submit that wrote them is known to be over (the next launch, a sync, or
    // the next use of the staging buffer).
    struct Readback { void* dst = nullptr; uint64_t bytes = 0; };
    std::vector<Readback> pending_readback;

    std::vector<std::unique_ptr<StreamImpl>> streams;
    StreamImpl* legacy = nullptr;
    std::map<std::string, Pipeline> pipes;
    Pipeline* noop_pipe = nullptr;

    bool lost = false;
};

// ------------------------------------------------------------------ the context: instance, selection, opening

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    bool tried = false;
    bool ok = false;
    std::string why;                          // why there is no usable device, when there is none
    std::vector<std::unique_ptr<Device>> devs;
    std::vector<std::string> skipped;         // what the selection dropped, with the reason it gave
    uint64_t max_import_align = 4096;         // the strictest minImportedHostPointerAlignment in the set
};
namespace {


// One VkQueue is not thread-safe and the engine does use more than one host thread (src/core/verify.cpp's side
// streams, src/core/overlap*.cpp), so every Vulkan call and every allocation lookup in this file goes through one
// recursive lock.  The engine's ORDER is still its own - streams and timeline values are what order its work - this
// lock only keeps two host threads off the same queue and off the same bookkeeping.
std::recursive_mutex g_gate;

std::atomic<Error> g_last_error{kSuccess};
thread_local int t_current_device = 0;

Error fail(Error e) {
    if (e != kSuccess && e != kNotReady) g_last_error.store(e, std::memory_order_relaxed);
    return e;
}

// One line per message name.  A runtime that reports the same problem 200 000 times is unreadable; one that reports
// it quietly is worse than one that does not run at all.
std::map<std::string, int> g_notes;
void note(const char* name, const char* fmt, ...) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (g_notes[std::string(name)]++ > 0) return;
    std::fprintf(stderr, "strata vulkan: ");
    // va_start / va_end are *macros* in MSVC's <stdarg.h>: writing them qualified expands std::va_start(ap, fmt) into
    // std::((void)(...)) and cl.exe answers C2589 "'(': illegal token on right side of '::'".  Unqualified is what
    // the macros are for.
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fflush(stderr);
}

bool env_on(const char* name, bool dflt = true) {           // an environment switch, ON unless it is spelled 0
    const char* e = std::getenv(name);
    if (e == nullptr) return dflt;
    return !(e[0] == '0' && e[1] == '\0');
}

uint64_t round_up(uint64_t v, uint64_t a) { return (a == 0 || a == 1) ? v : (v + a - 1) & ~(a - 1); }

const char* vk_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST (TDR or a driver reset)";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        default: return "another VkResult";
    }
}

void* aligned_host_alloc(size_t bytes, size_t align) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes ? bytes : 1, align < sizeof(void*) ? sizeof(void*) : align);
#else
    void* p = nullptr;
    const size_t a = align < sizeof(void*) ? sizeof(void*) : align;
    if (posix_memalign(&p, a, bytes ? bytes : 1) != 0) return nullptr;
    return p;
#endif
}

void aligned_host_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// A host allocation this port imported into a device stays RESIDENT after cudaHostFree: the range goes onto a free
// list of this port's own and the next cudaHostAlloc takes it back, instead of the pages going to the C allocator.
// What is measured and what is not, because the first version of this note claimed more than its transcripts showed
// (the re-measurement is vulkan/PHASE1-S5-VALIDATION-EVIDENCE.md): the losses ARE recorded, in 25 transcripts under
// build-vulkan/logs - 0/6 with no import, 0/4 with the import never torn down, 0/4 with the pages kept, 2/4 and 2/3
// with the pages released, 4/5 with a bounded drain first - always the A770, never the B580, no Windows TDR event.
// What is NOT established is that the validation layer was in those runs: not one of them carries a loader line, and
// the re-measurement produced no failing run to inspect - 24 driver-level runs of the pages-released shape lost
// nothing (vulkan/probe/vk_host_free_repro.cpp, 0 device losses in build-vulkan/logs/s5-repro-*.log).  The pool stays
// because it is the only change that coincided with the losses stopping, it costs no peak memory, and vkFreeMemory of
// an imported VkDeviceMemory is not specified to unmap those pages synchronously.
std::map<uint64_t, size_t> g_host_capacity;                    // pointer -> the capacity it was handed out with
std::vector<std::pair<void*, size_t>> g_host_free_list;        // retired (pointer, capacity), kept mapped and reused

void* host_pages_acquire(size_t bytes, size_t align, size_t& capacity_out) {
    size_t best = (size_t) -1, best_i = (size_t) -1;
    for (size_t i = 0; i < g_host_free_list.size(); ++i) {            // smallest range that is big enough
        const size_t cap = g_host_free_list[i].second;
        if (cap >= bytes && cap < best) { best = cap; best_i = i; }
    }
    if (best_i != (size_t) -1) {
        void* p = g_host_free_list[best_i].first;
        g_host_free_list.erase(g_host_free_list.begin() + (long) best_i);
        capacity_out = best;
        g_host_capacity[(uint64_t) (uintptr_t) p] = best;
        return p;
    }
    void* p = aligned_host_alloc(bytes ? bytes : 1, align);
    if (p == nullptr) { capacity_out = 0; return nullptr; }
    capacity_out = bytes ? bytes : 1;
    g_host_capacity[(uint64_t) (uintptr_t) p] = capacity_out;
    return p;
}

void host_pages_release(void* p) {
    if (p == nullptr) return;
    auto it = g_host_capacity.find((uint64_t) (uintptr_t) p);
    if (it == g_host_capacity.end()) { aligned_host_free(p); return; }   // not a range this port handed out
    const size_t cap = it->second;
    g_host_capacity.erase(it);
    g_host_free_list.push_back(std::make_pair(p, cap));
}


Ctx& ctx();
StreamImpl* make_stream(Device& d, uint32_t qidx, unsigned flags);
void read_budget(Device& d);
void selfcheck(Ctx& c);

// The predicate that picks this port's devices, in words: a DISCRETE GPU on an Intel driver.  It is a test on each
// enumerated physical device, NOT a position in vkEnumeratePhysicalDevices' array - the order that array comes back in
// is the loader's business, and on this PC the UHD 770 sits among the Arcs in it.  The gate-0 run made the same
// decision the same way (bench/results/vulkan-caps-gate-run-1.log:3-5):
//     physical devices: 3
//       skipping Intel(R) UHD Graphics 770 (integrated)
//     cards in this run: 2
bool choose_physical(const VkPhysicalDeviceProperties& p, const VkPhysicalDeviceDriverProperties& drv, std::string& why) {
    // 1. Integrated GPUs (the UHD 770) are out.  Vulkan names it with deviceType, and that is the property the machine
    //    rule is really about: an integrated card's "memory" is the system's RAM, so a per-card VRAM budget means
    //    nothing for it, and it is not the card the engine would put layers on.
    if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        why = "deviceType is not VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU";
        return false;
    }
    // 2. Intel identification from the device itself and from the driver's own report (VK_KHR_driver_properties), not
    //    from a name string: vendorID 0x8086 plus an Intel driverID.  Phase 0 P1 read exactly these off both Arcs
    //    (bench/results/vulkan-caps-gate-run-1.log:9-10, 45-46): vendor 8086, device 56a0 (A770) and e20b (B580),
    //    driver "Intel Corporation" id 5 "101.9150".
    if (p.vendorID != 0x8086) { why = "vendorID is not Intel's 0x8086"; return false; }
    if (drv.driverID != VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS && drv.driverID != VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) {
        why = "VK_KHR_driver_properties reports a driverID that is not an Intel driver";
        return false;
    }
    return true;
}

bool ext_available(VkPhysicalDevice pd, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> av(n ? n : 1);
    if (n) vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, av.data());
    for (const auto& x : av) if (std::strcmp(x.extensionName, name) == 0) return true;
    return false;
}

// The device-local heap: the one the budget numbers and the spill rule are about.  A770 heap 0 (16258 MiB, budget
// 15489), B580 heap 0 (12118 MiB, budget 11349) - bench/results/vulkan-caps-2026-10-17.md P1.
uint32_t device_local_heap(const Device& d) {
    int64_t best = -1;
    uint32_t idx = 0;
    for (uint32_t i = 0; i < d.mem.memoryHeapCount; ++i) {
        if (!(d.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
        if ((int64_t) d.mem.memoryHeaps[i].size > best) { best = (int64_t) d.mem.memoryHeaps[i].size; idx = i; }
    }
    return idx;
}

// The clamp every block, and every single storage-buffer binding, has to respect.  On both Arcs it is 4095 MiB:
// maxMemoryAllocationSize and maxStorageBufferRange are 4095 MiB and maxBufferSize is 4096 MiB (P1), so the smallest of
// the three is what binds - the largest allocation the driver advertises, the largest buffer it advertises, and the
// largest range one descriptor may cover.
uint64_t block_limit(const Device& d) {
    uint64_t lim = d.props.limits.maxStorageBufferRange;
    if (d.has_m4 && d.m4.maxBufferSize != 0 && d.m4.maxBufferSize < lim) lim = d.m4.maxBufferSize;
    if (d.m3.maxMemoryAllocationSize != 0 && d.m3.maxMemoryAllocationSize < lim) lim = d.m3.maxMemoryAllocationSize;
    if (lim == 0) lim = 4095ull * kMiB;
    return lim;
}

// STRATA_VULKAN_IGNORE_ALLOC_LIMIT=1 asks for the bench's "simple path" instead of the conformant split: P2 measured
// that this driver answers a 5120 MiB vkAllocateMemory even though it flags it against maxMemoryAllocationSize.  The
// default is the conformant one, which is what the plan's finding 4 says a portable port must do.
bool ignore_alloc_limit() {
    static const bool on = env_on("STRATA_VULKAN_IGNORE_ALLOC_LIMIT", false);
    return on;
}

bool find_mem(const Device& d, uint32_t bits, VkMemoryPropertyFlags want, uint32_t& out) {
    for (uint32_t i = 0; i < d.mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (d.mem.memoryTypes[i].propertyFlags & want) == want) { out = i; return true; }
    return false;
}

// The offset a sub-allocation gets inside a block: rounded up so the DEVICE POINTER the engine does its arithmetic on
// lands on `align` - 64, the minStorageBufferOffsetAlignment both Arcs report (P1) - because a VkDescriptorBufferInfo
// offset must be a multiple of that and a kernel's element indexing on `base + off` must be aligned too.  When the
// block's own base address is aligned (the driver gives page-aligned device addresses; P2 read 1.8e19-ish ones on both
// cards) the binding offset lands on the same 64.
uint64_t place_offset(uint64_t base_addr, uint64_t used, uint64_t align) {
    if (align == 0 || align == 1) return used;
    const uint64_t shift = (align - (base_addr % align)) % align;      // what this block's address needs first
    return round_up(used + shift, align);                              // ... then the allocation's own alignment
}

// Every block, host or device, is allocated with a device address on it: without
// VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT a buffer created with SHADER_DEVICE_ADDRESS has no address at all,
// vkGetBufferDeviceAddress returns 0, and a shader handed that pointer faults - vulkan/probe/vk_caps_probe.cpp:216-220
// says the same, and P2 (120 MiB spread fine, 2496 MiB spread lost the device, 4864 MiB spread clean) is the
// measurement that only exists because of that bit.
constexpr VkBufferUsageFlags kBlockUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

// The shape WITHOUT the device-address bit is kept only as documentation, because it is what the Phase 0 probe
// measured (vulkan/probe/vk_caps_probe.cpp:772-803: plain VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, no
// VkMemoryAllocateFlagsInfo) and what this port used while the intermittent VK_ERROR_DEVICE_LOST on the A770 was being
// tracked down.  That turned out to be about releasing the imported PAGES, not about this bit (see the note on
// host_pages_acquire), so the import below uses kBlockUsage: the device-address flag stays on every allocation this
// runtime makes, and a host allocation gets a real device address, which is what cudaHostGetDevicePointer implies.  If
// a driver ever objects to the combination, the switch is three edits in import_block - use
// `kBlockUsage & ~VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` for bi.usage, drop the VkMemoryAllocateFlagsInfo chained
// onto `imp`, and leave `b->addr` at the host address instead of asking vkGetBufferDeviceAddress.
VkResult make_block(Device& d, uint64_t bytes, VkMemoryPropertyFlags want, Block*& out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = kBlockUsage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(d.dev, &bi, nullptr, &buf);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d.dev, buf, &req);
    uint32_t mt = 0;
    if (!find_mem(d, req.memoryTypeBits, want, mt)) { vkDestroyBuffer(d.dev, buf, nullptr); return VK_ERROR_FEATURE_NOT_PRESENT; }

    const uint64_t alloc_size = round_up(std::max<uint64_t>(req.size, bytes), std::max<uint64_t>(req.alignment, 1));
    VkMemoryAllocateFlagsInfo af{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    af.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;               // EVERY allocation in this runtime carries one
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &af;
    ai.allocationSize = alloc_size;
    ai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    r = vkAllocateMemory(d.dev, &ai, nullptr, &mem);
    if (r != VK_SUCCESS) { vkDestroyBuffer(d.dev, buf, nullptr); return r; }
    r = vkBindBufferMemory(d.dev, buf, mem, 0);
    if (r != VK_SUCCESS) { vkFreeMemory(d.dev, mem, nullptr); vkDestroyBuffer(d.dev, buf, nullptr); return r; }

    std::unique_ptr<Block> b(new Block);
    b->dev = &d;
    b->mem = mem;
    b->buf = buf;
    b->size = alloc_size;
    b->mem_type = mt;
    b->props = d.mem.memoryTypes[mt].propertyFlags;
    if (d.get_address) {
        VkBufferDeviceAddressInfo si{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        si.buffer = buf;
        b->addr = d.get_address(d.dev, &si);
    }
    if (b->props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {           // mapped once, for the block's life
        void* m = nullptr;
        if (vkMapMemory(d.dev, mem, 0, b->size, 0, &m) == VK_SUCCESS) b->map = m;
    }
    out = b.get();
    d.blocks.push_back(std::move(b));
    return VK_SUCCESS;
}

// VK_EXT_external_memory_host, the route Phase 0 P3 measured (one host allocation, imported into both cards, correct
// in place, 4 KiB import alignment): the BUFFER carries
// VkExternalMemoryBufferCreateInfo::handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, the memory
// type is the one vkGetMemoryHostPointerPropertiesEXT reports for that exact host pointer, and the allocation imports
// the pages with VkImportMemoryHostPointerInfoEXT.  cudaHostAlloc(Mapped|Portable), cudaHostRegister and
// cudaHostGetDevicePointer are answered with this, which is what makes the alias those three calls promise real on a
// discrete card (src/prefill/prefill.cpp:1176-1177, src/core/expert_source.cpp:2897-2901, :2992-2995).
VkResult import_block(Device& d, void* ptr, uint64_t bytes, Block*& out) {
    if (!d.get_host_memory || !d.has_host_import) return VK_ERROR_EXTENSION_NOT_PRESENT;
    const uint64_t align = std::max<uint64_t>(1, d.hostp.minImportedHostPointerAlignment);   // 4 KiB on both Arcs
    const uint64_t lo = ((uint64_t) (uintptr_t) ptr) & ~(align - 1);
    const uint64_t slack = ((uint64_t) (uintptr_t) ptr) - lo;
    const uint64_t size = round_up(slack + bytes, align);           // whole granules of the import alignment

    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    const VkResult q = d.get_host_memory(d.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                         (const void*) (uintptr_t) lo, &hp);
    if (q != VK_SUCCESS) return q;                                  // the memory type this pointer may use
    uint32_t mt = 0;
    if (!find_mem(d, hp.memoryTypeBits, kHostWant, mt)) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkExternalMemoryBufferCreateInfo ex{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;   // ON THE BUFFER, per Phase 0 finding 5
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &ex;
    bi.size = size;
    bi.usage = kBlockUsage;                       // the same usage as any other block, device-address bit included:
                                                  // an imported host allocation is handed a device address too, which
                                                  // is what cudaHostGetDevicePointer is for.  The shape without the
                                                  // bit is described in the note above kBlockUsage, if a driver ever
                                                  // objects to the combination.
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(d.dev, &bi, nullptr, &buf);
    if (r != VK_SUCCESS) return r;

    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = (void*) (uintptr_t) lo;
    // VUID-vkBindBufferMemory-bufferDeviceAddress-03339: the memory behind a buffer that has SHADER_DEVICE_ADDRESS
    // must have been allocated with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, and this buffer has that usage, so the flag
    // rides on the import allocation too - chained after the import struct on the same VkMemoryAllocateInfo.  Every
    // allocation this runtime makes carries it, on the sub-allocator path and on this one.  (Phase 0's probe imported
    // without it, vulkan/probe/vk_caps_probe.cpp:772-803; neither shape changed the outcome - the note on
    // host_pages_acquire and PHASE1-S5-VALIDATION-EVIDENCE.md §6 say what that investigation did and did not show.)
    VkMemoryAllocateFlagsInfo af{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    af.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    imp.pNext = &af;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &imp;
    ai.allocationSize = round_up(size, std::max<uint64_t>(align, 1));
    ai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    r = vkAllocateMemory(d.dev, &ai, nullptr, &mem);
    if (r != VK_SUCCESS) { vkDestroyBuffer(d.dev, buf, nullptr); return r; }
    r = vkBindBufferMemory(d.dev, buf, mem, 0);
    if (r != VK_SUCCESS) { vkFreeMemory(d.dev, mem, nullptr); vkDestroyBuffer(d.dev, buf, nullptr); return r; }

    std::unique_ptr<Block> b(new Block);
    b->dev = &d;
    b->mem = mem;
    b->buf = buf;
    b->size = ai.allocationSize;
    b->used = b->size;                    // an import is handed out whole, never sub-allocated
    b->live = bytes;
    b->mem_type = mt;
    b->props = d.mem.memoryTypes[mt].propertyFlags;
    b->imported = true;
    b->host_ptr = (void*) (uintptr_t) lo;
    b->map = b->host_ptr;                 // the host's own pages ARE the CPU view of this buffer
    // The device pointer of an imported host allocation, asked from the buffer the way every other block's is asked.
    // If the driver hands back 0 (or the device lacks the address entry point) the host address is used instead, which
    // is what CUDA's unified addressing reports for a mapped host allocation anyway - the pages on both sides are the
    // same pages either way, and host_view() maps this block back to them regardless.
    b->addr = (uint64_t) (uintptr_t) lo;
    if (d.get_address) {
        VkBufferDeviceAddressInfo si{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        si.buffer = buf;
        const VkDeviceAddress a = d.get_address(d.dev, &si);
        if (a != 0) b->addr = a;
    }
    out = b.get();
    d.blocks.push_back(std::move(b));
    return VK_SUCCESS;
}

// A sub-range of a block, or a new block when no live block has room.  Every offset handed out is rounded up to the
// card's minStorageBufferOffsetAlignment (64 on both Arcs, P1); the uniform parameter block uses the card's uniform
// alignment when that is coarser.
Alloc* sub_alloc(Device& d, uint64_t bytes, VkMemoryPropertyFlags want, uint64_t align, bool* out_of_memory) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (out_of_memory) *out_of_memory = false;
    if (bytes == 0) bytes = 1;
    if (align == 0) align = 1;

    std::unique_ptr<Alloc> a(new Alloc);
    a->dev = &d;
    a->bytes = bytes;
    a->align = align;

    // First fit in the live blocks of this flavour, holes included, then a new block.
    for (auto& bp : d.blocks) {
        if (bp->imported || bp->buf == VK_NULL_HANDLE || (bp->props & want) != want) continue;
        const uint64_t cap = ignore_alloc_limit() ? bp->size : std::min<uint64_t>(bp->size, block_limit(d));
        if (bytes > cap) continue;
        for (size_t hi = 0; hi < bp->holes.size(); ++hi) {
            Hole& h = bp->holes[hi];
            const uint64_t off = place_offset(bp->addr, h.off, align);     // rounded up to the card's alignment
            if (off + bytes <= h.off + h.bytes) {
                Span s{bp.get(), off, bp->addr + off, bytes};
                a->spans.push_back(s);
                a->addr = s.addr;
                h.off = off + bytes;
                h.bytes -= bytes;
                if (h.bytes == 0) bp->holes.erase(bp->holes.begin() + (long) hi);
                bp->live += bytes;
                d.allocs.push_back(std::move(a));
                return d.allocs.back().get();
            }
        }
        const uint64_t off = place_offset(bp->addr, bp->used, align);      // rounded up to the card's alignment
        if (off + bytes <= cap) {
            Span s{bp.get(), off, bp->addr + off, bytes};
            a->spans.push_back(s);
            a->addr = s.addr;
            bp->used = off + bytes;
            bp->live += bytes;
            d.allocs.push_back(std::move(a));
            return d.allocs.back().get();
        }
    }

    const uint64_t lim = block_limit(d);
    if (!ignore_alloc_limit() && bytes > lim) {
        // One allocation bigger than the clamp is only handed out if the blocks that serve it come out contiguous in
        // device-address space, because the engine's device pointers ARE arithmetic (cache_base + slot_off, the ptr2
        // tables, the ptr3d pitch rows).  If the driver does not place them contiguously this is refused with the
        // numbers in the message rather than answered with a pointer that would read wrong data.
        uint64_t left = bytes;
        while (left != 0) {
            const uint64_t take = std::min<uint64_t>(left, lim);
            Block* blk = nullptr;
            VkResult r = make_block(d, take, want, blk);
            if (r != VK_SUCCESS && (want & kDeviceWant) == kDeviceWant)
                r = make_block(d, take, kDeviceWantStrict, blk);   // a card that does not map all of its VRAM
            if (r != VK_SUCCESS) {
                if (out_of_memory) *out_of_memory = true;
                note("oversize", "a %llu MiB allocation needs %d blocks because this port clamps one block to "
                                 "min(maxMemoryAllocationSize, maxBufferSize, maxStorageBufferRange) = %llu MiB "
                                 "(bench/results/vulkan-caps-2026-10-17.md P1); vkAllocateMemory said %s.  P2 measured "
                                 "that this driver answers one block over that limit, so STRATA_VULKAN_IGNORE_ALLOC_"
                                 "LIMIT=1 asks for the bench's simple path instead of the conformant split.\n",
                     (unsigned long long) (bytes / kMiB), (int) (bytes / lim) + 1, (unsigned long long) (lim / kMiB),
                     vk_name(r));
                return nullptr;
            }
            a->spans.push_back(Span{blk, 0, blk->addr, take});
            blk->used = take;
            blk->live = take;
            left -= take;
        }
        for (size_t i = 1; i < a->spans.size(); ++i) {
            if (a->spans[i].addr != a->spans[i - 1].addr + a->spans[i - 1].bytes) {
                note("contiguity", "the %zu blocks serving a %llu MiB allocation did not come out contiguous in "
                                   "device-address space (%llu then %llu), so no single pointer can address the whole "
                                   "range.  Split the region in the caller, or run with STRATA_VULKAN_IGNORE_ALLOC_"
                                   "LIMIT=1 (Phase 0 P2: this driver answers a 5120 MiB block).  Each block was %llu "
                                   "MiB.\n", a->spans.size(), (unsigned long long) (bytes / kMiB),
                     (unsigned long long) a->spans[i - 1].addr, (unsigned long long) a->spans[i].addr,
                     (unsigned long long) (a->spans[0].bytes / kMiB));
                for (Span& s : a->spans) {
                    vkDestroyBuffer(d.dev, s.block->buf, nullptr);
                    vkFreeMemory(d.dev, s.block->mem, nullptr);
                    s.block->mem = VK_NULL_HANDLE; s.block->buf = VK_NULL_HANDLE; s.block->map = nullptr; s.block->live = 0;
                }
                a->spans.clear();
                if (out_of_memory) *out_of_memory = true;
                return nullptr;
            }
        }
        a->addr = a->spans.front().addr;
        d.allocs.push_back(std::move(a));
        return d.allocs.back().get();
    }

    uint64_t size = std::max<uint64_t>((want & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? kDeviceBlockBytes : kHostBlockBytes,
                                       bytes);
    size = std::min<uint64_t>(size, lim);                          // a block is never bigger than the clamp
    Block* blk = nullptr;
    VkResult r = make_block(d, size, want, blk);
    if (r != VK_SUCCESS && want == kDeviceWant)
        r = make_block(d, size, kDeviceWantStrict, blk);          // a card without a host-visible VRAM type
    if (r != VK_SUCCESS) { if (out_of_memory) *out_of_memory = true; return nullptr; }
    const uint64_t off = place_offset(blk->addr, blk->used, align);   // rounded up to the card's alignment
    a->spans.push_back(Span{blk, off, blk->addr + off, bytes});
    a->addr = a->spans.back().addr;
    blk->used = off + bytes;
    blk->live += bytes;
    d.allocs.push_back(std::move(a));
    return d.allocs.back().get();
}

const Span* span_at(const Alloc* a, uint64_t off) {
    for (const Span& s : a->spans)
        if (off >= (s.addr - a->addr) && off < (s.addr - a->addr) + s.bytes) return &s;
    return nullptr;
}

void register_alloc(Alloc* a) {
    a->dev->by_dev[a->addr] = a;
    if (a->host) a->dev->by_host[(uint64_t) (uintptr_t) a->host] = a;
    if (!a->host_alloc) a->dev->own_device_bytes += a->bytes;     // exactly what cudaMemGetInfo subtracts
}

void push_hole(Block& b, uint64_t off, uint64_t bytes) {
    if (bytes == 0) return;
    for (auto& h : b.holes) {
        if (h.off + h.bytes == off) { h.bytes += bytes; return; }
        if (off + bytes == h.off) { h.off = off; h.bytes += bytes; return; }
    }
    b.holes.push_back(Hole{off, bytes});
    std::sort(b.holes.begin(), b.holes.end(), [](const Hole& x, const Hole& y) { return x.off < y.off; });
}

void unregister_alloc(Alloc* a) {
    Device& d = *a->dev;
    d.by_dev.erase(a->addr);
    if (a->host) d.by_host.erase((uint64_t) (uintptr_t) a->host);
    if (!a->host_alloc) d.own_device_bytes = d.own_device_bytes > a->bytes ? d.own_device_bytes - a->bytes : 0;
    for (Span& s : a->spans) {
        Block* b = s.block;
        if (b->buf == VK_NULL_HANDLE) continue;
        b->live = b->live >= s.bytes ? b->live - s.bytes : 0;
        if (b->imported) continue;                        // an import is destroyed whole, not holed
        push_hole(*b, s.block_off, s.bytes);
        if (b->live == 0) {                               // nothing is left in it: give the VkDeviceMemory back
            vkDestroyBuffer(d.dev, b->buf, nullptr);
            vkFreeMemory(d.dev, b->mem, nullptr);
            b->buf = VK_NULL_HANDLE; b->mem = VK_NULL_HANDLE; b->map = nullptr;
        }
    }
}

void destroy_alloc(Device& d, Alloc* a) {
    if (a == nullptr) return;
    unregister_alloc(a);
    auto it = std::find_if(d.allocs.begin(), d.allocs.end(),
                           [&](const std::unique_ptr<Alloc>& x) { return x.get() == a; });
    if (it != d.allocs.end()) d.allocs.erase(it);
}

// The allocation holding `p` - a device address or a CPU address - with the offset into it.  Same shape as the Metal
// runtime's lookup (metal/src/gpu/runtime.cpp:139-155): an interval search, because what arrives is `base + off`,
// not only a base.
Alloc* find_alloc(Device& d, const void* p, uint64_t* off, bool* is_host) {
    const uint64_t a = (uint64_t) (uintptr_t) p;
    auto it = d.by_dev.upper_bound(a);
    if (it != d.by_dev.begin()) {
        --it;
        Alloc* al = it->second;
        if (a >= al->addr && a < al->addr + al->bytes) {
            if (off) *off = a - al->addr;
            if (is_host) *is_host = false;
            return al;
        }
    }
    auto ih = d.by_host.upper_bound(a);
    if (ih != d.by_host.begin()) {
        --ih;
        Alloc* al = ih->second;
        const uint64_t base = (uint64_t) (uintptr_t) al->host;
        if (a >= base && a < base + al->bytes) {
            if (off) *off = a - base;
            if (is_host) *is_host = true;
            return al;
        }
    }
    return nullptr;
}

Device* current_device(const char* what) {
    Ctx& c = ctx();
    if (!c.ok) {
        note("no-device", "%s: no usable Vulkan device in this process (%s)\n", what, c.why.c_str());
        return nullptr;
    }
    int i = t_current_device;
    if (i < 0 || (size_t) i >= c.devs.size()) i = 0;
    if (c.devs[(size_t) i]->lost) { note("lost", "%s: this device was lost earlier in the run\n", what); return nullptr; }
    return c.devs[(size_t) i].get();
}

// ------------------------------------------------------------------ streams: queues, chains, submits, timelines

// The host->GPU signal this driver does allow.  Phase 0 P5: a flag the host wrote was never observed by a dispatch that
// was already running, in any of four host-visible flavours and with both host write methods, so a host write becomes
// visible by being followed by a SUBMIT, never by being polled.  Every copy, fill, launch and event record ends with
// this, or with a timeline wait below.
void nudge_submit(StreamImpl* s);

// A chain slot is reused only after its submit is over AND its fence is put back to unsignaled: vkQueueSubmit requires
// an unsignaled fence (VUID-vkQueueSubmit-fence-00063 - "submitted in SIGNALED state. Fences must be reset before
// being submitted", which the layer said in build-vulkan/logs/s3-smoke-04.log:105 on the first replayed-graph submit).
void retire_chain(StreamImpl* s, Chain& c) {
    if (c.live) {
        // Bounded, not infinite: a submit that never finishes is a stall, and an unbounded wait on a stall is how a
        // runtime gets its device taken away by the driver's watchdog.  The timeout is named when it happens.
        if (vkWaitForFences(s->d->dev, 1, &c.fence, VK_TRUE, kFenceWaitNs) != VK_SUCCESS)
            note("fence", "a submit on stream %u of %s had not finished after %.1f s (a timeline wait whose signal "
                          "never arrives, or a driver that stopped retiring work)\n",
                 s->qidx, s->d->props.deviceName, (double) kFenceWaitNs / 1e9);
        c.live = false;
    }
    if (c.fence != VK_NULL_HANDLE) vkResetFences(s->d->dev, 1, &c.fence);
}

// One submit: the stream's open chain (or a captured graph's buffer), the waits the stream was told to take, a signal
// on the stream's own timeline, and a fence.  This is both how a stream's work is serialized and how an event's value
// is published.
VkResult submit_work(StreamImpl* s, VkCommandBuffer cb, VkFence fence) {
    Device& d = *s->d;
    std::vector<VkSemaphore> waits;
    std::vector<uint64_t> wait_values;
    std::vector<VkPipelineStageFlags> stages;
    for (const auto& w : s->waits) {
        waits.push_back(w.first);
        wait_values.push_back(w.second);
        stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    std::vector<VkSemaphore> sigs;
    std::vector<uint64_t> sig_values;
    const uint64_t tl_value = ++s->signaled;
    sigs.push_back(s->tl); sig_values.push_back(tl_value);          // every submit signals its stream's timeline
    for (const auto& g : s->signals) { sigs.push_back(g.first); sig_values.push_back(g.second); }

    VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    tsi.waitSemaphoreValueCount = (uint32_t) wait_values.size();
    tsi.pWaitSemaphoreValues = wait_values.empty() ? nullptr : wait_values.data();
    tsi.signalSemaphoreValueCount = (uint32_t) sig_values.size();
    tsi.pSignalSemaphoreValues = sig_values.data();

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &tsi;                                                // the timeline values belong to the submit
    si.waitSemaphoreCount = (uint32_t) waits.size();
    si.pWaitSemaphores = waits.empty() ? nullptr : waits.data();    // the WAIT entries of the submit
    si.pWaitDstStageMask = stages.empty() ? nullptr : stages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = (uint32_t) sigs.size();
    si.pSignalSemaphores = sigs.data();                             // the SIGNAL entries of the submit

    const VkResult r = vkQueueSubmit(s->q, 1, &si, fence);          // ... and every submit carries a fence
    s->waits.clear();
    s->signals.clear();
    if (r != VK_SUCCESS) {
        note("submit", "vkQueueSubmit on stream %u of %s said %s\n", s->qidx, d.props.deviceName, vk_name(r));
        if (r == VK_ERROR_DEVICE_LOST) d.lost = true;
        fail(kErrorLaunchFailure);
        return r;
    }
    s->submits++;
    return VK_SUCCESS;
}

// The GPU->host signal this driver does allow: a timeline semaphore value, waited on in slices so a lost device is
// reported instead of hung on.  No spin loop, no host flag, no polling shader.
Error wait_timeline(Device& d, VkSemaphore sem, uint64_t value, const char* what) {
    if (sem == VK_NULL_HANDLE) return kSuccess;
    const VkSemaphore* p = &sem;
    const uint64_t* v = &value;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = p;
    wi.pValues = v;
    for (;;) {
        const VkResult r = vkWaitSemaphores(d.dev, &wi, kWaitSliceNs);      // the host-side wait on a timeline value
        if (r == VK_SUCCESS) return kSuccess;
        if (r == VK_TIMEOUT) {
            if (d.lost) { note("wait-lost", "%s: the device was lost while waiting\n", what); return fail(kErrorLaunchFailure); }
            continue;
        }
        note("wait", "%s: vkWaitSemaphores said %s (waiting for value %llu on %s)\n", what, vk_name(r),
             (unsigned long long) value, d.props.deviceName);
        if (r == VK_ERROR_DEVICE_LOST) { d.lost = true; return fail(kErrorLaunchFailure); }
        return fail(kErrorLaunchFailure);
    }
}

Chain* open_chain(StreamImpl* s) {
    Chain& c = s->chain[s->chain_next];
    s->chain_next = (s->chain_next + 1) % kChainSlots;
    retire_chain(s, c);                       // at most kChainSlots in flight; reusing a slot waits for its fence
    if (vkResetCommandBuffer(c.cb, 0) != VK_SUCCESS) { note("cb-reset", "vkResetCommandBuffer failed on a chain slot"); return nullptr; }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(c.cb, &bi) != VK_SUCCESS) { note("cb-begin", "vkBeginCommandBuffer failed on a chain slot"); return nullptr; }
    c.recording = true;
    c.param_used = 0;
    s->open = &c;
    s->recorded = 0;
    return &c;
}

// The command buffer a recorded command goes into: while the stream is capturing that is the graph's single
// VkCommandBuffer, otherwise the stream's open chain slot.
VkCommandBuffer record_cb(StreamImpl* s) {
    if (s->rec) return s->rec->cb;
    if (s->open) return s->open->cb;
    Chain* c = open_chain(s);
    return c ? c->cb : VK_NULL_HANDLE;
}

void nudge_submit(StreamImpl* s) {
    if (s->rec) return;                       // a capture is submitted by cudaGraphLaunch, not here
    if (!s->open) {
        if (!s->signals.empty()) {            // nothing recorded, but a recorded event still has to be signalled
            Chain& c = s->chain[s->chain_next];
            s->chain_next = (s->chain_next + 1) % kChainSlots;
            retire_chain(s, c);
            if (vkResetCommandBuffer(c.cb, 0) != VK_SUCCESS) return;
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(c.cb, &bi) != VK_SUCCESS) return;
            if (vkEndCommandBuffer(c.cb) != VK_SUCCESS) return;
            submit_work(s, c.cb, c.fence);    // an empty submit is still the doorbell this driver takes
            c.live = true;
        }
        return;
    }
    Chain& c = *s->open;
    if (vkEndCommandBuffer(c.cb) != VK_SUCCESS) { note("cb-end", "vkEndCommandBuffer failed on a chain slot"); return; }
    submit_work(s, c.cb, c.fence);
    c.live = true;
    c.recording = false;
    s->open = nullptr;
    s->recorded = 0;
}

void flush_stream(StreamImpl* s) { nudge_submit(s); }

StreamImpl* stream_of(Device& d, Stream s) {
    if (s) return s;
    if (d.legacy == nullptr) d.legacy = make_stream(d, 0, 0);
    return d.legacy;
}

// The staging buffer a copy goes through when one side is host memory this port does not own (a plain malloc, an
// mmap'd arena, a file buffer): Vulkan transfers are buffer-to-buffer, there is no "copy from a malloc" command.
// HOST_VISIBLE|HOST_COHERENT, so a CPU write followed by a submit is enough - no flush, and no polling either.
void drain_readback(Device& d) {
    if (d.pending_readback.empty()) return;
    if (d.staging_pending && d.staging_stream != nullptr)
        wait_timeline(*d.staging_stream->d, d.staging_stream->tl, d.staging_value, "staged copy read-back");
    d.staging_pending = false;
    if (d.staging != nullptr) {
        for (const Device::Readback& r : d.pending_readback)
            if (r.dst != nullptr && r.bytes <= d.staging_bytes) std::memcpy(r.dst, d.staging->host, r.bytes);
    }
    d.pending_readback.clear();
}

Alloc* staging_for(Device& d, uint64_t need) {
    if (d.staging_pending) {                  // the last use of it has to be over before it is written again
        drain_readback(d);
        if (d.staging_stream != nullptr) wait_timeline(*d.staging_stream->d, d.staging_stream->tl, d.staging_value, "staging reuse");
        d.staging_pending = false;
    }
    if (d.staging != nullptr && d.staging_bytes >= need) return d.staging;
    bool oom = false;
    const uint64_t size = std::max<uint64_t>(need, kHostBlockBytes);
    Alloc* a = sub_alloc(d, size, kHostWant, d.align_storage, &oom);
    if (a == nullptr) { note("staging", "no host-visible staging buffer of %llu MiB (%s)\n",
                             (unsigned long long) (size / kMiB), oom ? "out of memory" : "no such memory type"); return nullptr; }
    a->host = (char*) a->spans[0].block->map + a->spans[0].block_off;   // the mapped CPU view of the same bytes
    register_alloc(a);
    d.staging = a;
    d.staging_bytes = size;
    return a;
}

struct Side {
    Block* block = nullptr;
    VkBuffer buf = VK_NULL_HANDLE;
    uint64_t off = 0;
    void* cpu = nullptr;      // the CPU view of this side, when it has one
    bool staged = false;      // true when this side is the staging buffer standing in for unknown host memory
};

bool side_of(Device& d, const void* p, uint64_t n, Side& out) {
    uint64_t aoff = 0;
    bool is_host = false;
    out = Side{};
    if (Alloc* a = find_alloc(d, p, &aoff, &is_host)) {
        const uint64_t base = is_host ? (uint64_t) (uintptr_t) a->host : a->addr;
        const uint64_t inside = (uint64_t) (uintptr_t) p - base;
        if (const Span* s = span_at(a, inside)) {
            out.block = s->block;
            out.buf = s->block->buf;
            out.off = s->block_off + inside;
            out.cpu = s->block->map != nullptr ? (char*) s->block->map + s->block_off + inside : nullptr;
            return out.buf != VK_NULL_HANDLE;
        }
    }
    Alloc* st = staging_for(d, n);
    if (st == nullptr) return false;
    out.block = st->spans[0].block;
    out.buf = out.block->buf;
    out.off = st->spans[0].block_off;
    out.cpu = st->host;
    out.staged = true;
    return true;
}

// One copy.  `into` is the command buffer it is recorded into: VK_NULL_HANDLE means the stream's own open chain, which
// is then submitted (and waited on when a read-back is needed); a non-null buffer means the caller submits it (a
// captured graph's buffer).  The copy is split where either side leaves its buffer.
Error copy_region(Device& d, StreamImpl* s, void* dst, const void* src, size_t n, CopyKind kind, bool sync,
                  VkCommandBuffer into = VK_NULL_HANDLE) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (n == 0) return kSuccess;
    if (kind == kHostToHost) { std::memcpy(dst, src, n); return kSuccess; }

    Side a, b;
    const bool dst_known = side_of(d, dst, n, a) && !a.staged;
    const bool src_known = side_of(d, src, n, b) && !b.staged;
    if (!dst_known && !src_known) {
        note("copy", "neither side of a %llu B copy is a buffer this runtime knows (dst %p, src %p), and Vulkan has no "
                     "copy between two host addresses it has no buffer for\n", (unsigned long long) n, dst, src);
        return fail(kErrorInvalidValue);
    }
    if (!src_known) std::memcpy(b.cpu, src, n);           // host memory -> the mapped staging buffer
    VkCommandBuffer cb = into != VK_NULL_HANDLE ? into : record_cb(s);
    if (cb == VK_NULL_HANDLE) return fail(kErrorInvalidValue);
    uint64_t done = 0;
    while (done < n) {
        // One region per run that stays inside both buffers.  A VkCmdCopyBuffer region must be a whole number of
        // 4-byte words (VUID-vkCmdCopyBuffer-pRegions-00188) at 4-byte-aligned offsets (the block offsets are 64-aligned
        // by sub_alloc, which is what makes that true), so a size like 37 B is rounded up to 40 and the extra bytes land
        // in the block's own slack: every allocation offset is a multiple of 64, so there is always room past an
        // allocation that is not itself a multiple of 4.
        const uint64_t logical = std::min<uint64_t>(n - done,
            std::min<uint64_t>(a.block->size - a.off - done, b.block->size - b.off - done));
        if (logical == 0) {
            note("copy-fit", "a %llu B copy does not fit in the buffers it was given (dst %p in a %llu B block at "
                             "offset %llu, src %p in a %llu B block at offset %llu)\n", (unsigned long long) n, dst,
                 (unsigned long long) a.block->size, (unsigned long long) a.off, src,
                 (unsigned long long) b.block->size, (unsigned long long) b.off);
            return fail(kErrorInvalidValue);
        }
        const uint64_t rounded = round_up(logical, 4);
        if (rounded != logical && (a.off + done + rounded > a.block->size || b.off + done + rounded > b.block->size)) {
            note("copy-words", "the trailing %llu B of a %llu B copy cannot be expressed as a whole 4-byte region "
                               "inside both buffers (VUID-vkCmdCopyBuffer-pRegions-00188), and neither side has the 4 "
                               "bytes of block slack to round up into.  Sizes this port can copy exactly are multiples "
                               "of 4, or allocations with slack behind them.\n", (unsigned long long) (logical % 4),
                 (unsigned long long) n);
            return fail(kErrorInvalidValue);
        }
        VkBufferCopy cp{b.off + done, a.off + done, rounded};   // srcOffset in b (the source side), dstOffset in a (the
        vkCmdCopyBuffer(cb, b.buf, a.buf, 1, &cp);              // destination side).  vkCmdCopyBuffer's own argument
                                                                // order is (srcBuffer, dstBuffer): the side named `a`
                                                                // here is the DESTINATION, so the two must not be
                                                                // swapped - that was the first version's bug.
        done += logical;
    }
    s->copies++;
    if (s->rec != nullptr) {                         // a CUDA capture records EVERY op as a node, and a replay
        NodeImpl* nd = new NodeImpl;                 // re-encodes the node list: a copy recorded into a capture that
        nd->type = (int) cudaGraphNodeTypeMemcpy;    // never became a node would vanish on the second launch
        nd->dst = dst; nd->src = src; nd->bytes = n;
        s->rec->nodes.push_back(nd);
    }
    if (into != VK_NULL_HANDLE) {                         // the caller submits; only a read-back is deferred
        if (!dst_known) { d.pending_readback.push_back(Device::Readback{dst, n}); d.staging_pending = true;
                          d.staging_stream = s; d.staging_value = s->signaled; }
        return kSuccess;
    }
    nudge_submit(s);                                      // the write is only visible to the GPU after this
    if (sync || !dst_known) {
        wait_timeline(d, s->tl, s->signaled, "copy");    // ... and the read-back only after that submit is done
        if (!dst_known) std::memcpy(dst, a.cpu, n);      // the mapped staging buffer -> the host's own memory
    }
    d.staging_pending = false;
    return kSuccess;
}

Error fill_region(Device& d, StreamImpl* s, void* p, int value, size_t n, bool sync,
                  VkCommandBuffer into = VK_NULL_HANDLE) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (n == 0) return kSuccess;
    Side a;
    if (!side_of(d, p, n, a)) { note("fill", "a fill target is not a buffer this runtime knows (%p)", p); return fail(kErrorInvalidValue); }
    const uint64_t b = (uint64_t) (value & 0xff);
    const uint64_t pattern = b | (b << 8) | (b << 16) | (b << 24);   // CUDA's memset writes BYTES, vkCmdFillBuffer a word
    const uint64_t rounded = round_up(n, 4);   // vkCmdFillBuffer also wants whole 4-byte words (VUID-vkCmdFillBuffer-size-00010)
    if (a.off + rounded > a.block->size) {
        note("fill-fit", "a %llu B fill does not fit, with its 4-byte rounding, in the %llu B block holding %p at "
                         "offset %llu\n", (unsigned long long) n, (unsigned long long) a.block->size, p,
             (unsigned long long) a.off);
        return fail(kErrorInvalidValue);
    }
    VkCommandBuffer cb = into != VK_NULL_HANDLE ? into : record_cb(s);
    if (cb == VK_NULL_HANDLE) return fail(kErrorInvalidValue);
    vkCmdFillBuffer(cb, a.buf, a.off, rounded, (uint32_t) pattern);
    s->fills++;
    if (s->rec != nullptr) {                         // the same node rule as a captured copy (see copy_region)
        NodeImpl* nd = new NodeImpl;
        nd->type = (int) cudaGraphNodeTypeMemset;
        nd->dst = p; nd->bytes = n; nd->value = value;
        s->rec->nodes.push_back(nd);
    }
    if (into == VK_NULL_HANDLE) { nudge_submit(s); if (sync) wait_timeline(d, s->tl, s->signaled, "memset"); }
    return kSuccess;
}

// A descriptor binding for one span.  The RANGE is what the card clamps; the OFFSET is not.
// Phase 0 measured exactly this on both Arcs (bench/results/vulkan-caps-2026-10-17.md:107-115): no binding may cover
// more than maxStorageBufferRange (4095 MiB), while a descriptor offset may sit anywhere, including past 4 GiB -
// P2b bound offset 4096 MiB with range 512 MiB and read and wrote correct tags on both cards, and a device address
// reached 5056 MiB inside one allocation.  So the test below is on `bytes` and deliberately NOT on `off`: the engine's
// layout rule is "keep each binding under 4095 MiB, use offsets freely, including past 4 GiB".
bool bind_span(Device& d, VkBuffer buf, uint64_t off, uint64_t bytes, VkDescriptorBufferInfo& out) {
    out.buffer = buf;
    out.offset = off;
    out.range = bytes == 0 ? 1 : bytes;
    if (bytes > block_limit(d)) {
        note("bind-range", "a storage-buffer binding of %llu MiB over %s exceeds this card's "
                           "min(maxStorageBufferRange, maxBufferSize, maxMemoryAllocationSize) = %llu MiB.  Its OFFSET "
                           "(%llu MiB) is not limited and stays as it is - Phase 0 P2b: a 512 MiB range at offset "
                           "4096 MiB read and wrote correct tags on both cards.  Split the binding in the caller.\n",
             (unsigned long long) (bytes / kMiB), d.props.deviceName, (unsigned long long) (block_limit(d) / kMiB),
             (unsigned long long) (off / kMiB));
        return false;
    }
    if (off % d.align_storage) {
        note("bind-align", "a descriptor offset of %llu on %s is not a multiple of minStorageBufferOffsetAlignment "
                           "%llu (the block's own base address is what made it so)\n", (unsigned long long) off,
             (unsigned long long) d.align_storage, d.props.deviceName);
    }
    return true;
}

// ------------------------------------------------------------------ pipelines and dispatch (where Phase 2 plugs in)

Pipeline* find_pipeline(Device& d, const char* name) {
    auto it = d.pipes.find(name ? name : "");
    return it == d.pipes.end() ? nullptr : &it->second;
}

Error build_pipeline(Device& d, const PipelineSpec& spec, std::string& why) {
    if (spec.spv == nullptr || spec.spv_words == 0) { why = "no SPIR-V"; return fail(kErrorInvalidValue); }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = spec.spv_words * 4;
    mi.pCode = spec.spv;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(d.dev, &mi, nullptr, &mod) != VK_SUCCESS) { why = "vkCreateShaderModule failed"; return fail(kErrorInvalidValue); }

    Pipeline p;
    p.name = spec.name ? spec.name : "";
    p.key = p.name;
    p.entry = (spec.entry && spec.entry[0]) ? spec.entry : "main";
    p.mod = mod;
    p.local[0] = spec.workgroup.x ? spec.workgroup.x : 1;
    p.local[1] = spec.workgroup.y ? spec.workgroup.y : 1;
    p.local[2] = spec.workgroup.z ? spec.workgroup.z : 1;
    p.shared = (uint32_t) spec.shared_mem;
    p.param_bytes = (uint32_t) spec.param_bytes;
    p.storage_count = spec.storage_bindings;
    p.push_params = p.param_bytes != 0 && p.param_bytes <= d.props.limits.maxPushConstantsSize;   // 256 B on both Arcs,
                                                                                                    // measured by the
                                                                                                    // self-check dump
    p.param_binding = 0;

    // The plan's hard rule is that launch parameters travel as VkPushConstantRange / vkCmdPushConstants, not in a
    // uniform buffer.  So a kernel whose parameter block is bigger than this card's maxPushConstantsSize is REFUSED at
    // registration by default, named, rather than silently given a uniform-buffer parameter block: the numbers are
    // small (a CUDA kernel's arguments are a handful of pointers and scalars, and both Arcs answer 256 B), and a
    // kernel that really needs more has to say so.  STRATA_VULKAN_UNIFORM_PARAMS=1 selects the uniform-buffer path
    // below instead, for whoever decides that trade is worth it.
    if (!p.push_params && p.param_bytes != 0 && !env_on("STRATA_VULKAN_UNIFORM_PARAMS", false)) {
        note("param-size", "\"%s\" asks for %llu bytes of launch parameters, which is more than %s reports as "
                           "maxPushConstantsSize = %llu B.  This port passes launch parameters with vkCmdPushConstants "
                           "by design; register the kernel with a parameter block that fits, or set "
                           "STRATA_VULKAN_UNIFORM_PARAMS=1 to take the uniform-buffer path instead.\n",
             p.name.c_str(), (unsigned long long) p.param_bytes, d.props.deviceName,
             (unsigned long long) d.props.limits.maxPushConstantsSize);
        vkDestroyShaderModule(d.dev, mod, nullptr);
        return fail(kErrorInvalidValue);
    }

    uint32_t idx = 0;
    if (!p.push_params && p.param_bytes != 0) {
        VkDescriptorSetLayoutBinding bd{};
        bd.binding = idx++;
        bd.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bd.descriptorCount = 1;
        bd.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        p.param_binding = bd.binding;
        p.bindings.push_back(bd);
    }
    for (uint32_t i = 0; i < p.storage_count; ++i) {         // one storage binding per device-address parameter
        VkDescriptorSetLayoutBinding bd{};
        bd.binding = idx++;
        bd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bd.descriptorCount = 1;
        bd.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        p.bindings.push_back(bd);
    }
    if (!p.bindings.empty()) {
        VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        di.bindingCount = (uint32_t) p.bindings.size();
        di.pBindings = p.bindings.data();
        if (vkCreateDescriptorSetLayout(d.dev, &di, nullptr, &p.dsl) != VK_SUCCESS) {
            vkDestroyShaderModule(d.dev, mod, nullptr); why = "vkCreateDescriptorSetLayout failed"; return fail(kErrorInvalidValue);
        }
        std::vector<VkDescriptorPoolSize> sizes;
        if (!p.push_params && p.param_bytes) sizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kChainSlots * 4});
        if (p.storage_count) sizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, p.storage_count * kChainSlots * 4});
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = kChainSlots * 4;
        pi.poolSizeCount = (uint32_t) sizes.size();
        pi.pPoolSizes = sizes.data();
        if (vkCreateDescriptorPool(d.dev, &pi, nullptr, &p.pool) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(d.dev, p.dsl, nullptr); vkDestroyShaderModule(d.dev, mod, nullptr);
            why = "vkCreateDescriptorPool failed"; return fail(kErrorInvalidValue);
        }
    }

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (p.dsl != VK_NULL_HANDLE) { li.setLayoutCount = 1; li.pSetLayouts = &p.dsl; }
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, p.param_bytes};
    if (p.push_params) { li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr; }
    if (vkCreatePipelineLayout(d.dev, &li, nullptr, &p.layout) != VK_SUCCESS) {
        if (p.pool) vkDestroyDescriptorPool(d.dev, p.pool, nullptr);
        if (p.dsl) vkDestroyDescriptorSetLayout(d.dev, p.dsl, nullptr);
        vkDestroyShaderModule(d.dev, mod, nullptr);
        why = "vkCreatePipelineLayout failed"; return fail(kErrorInvalidValue);
    }

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = p.mod;
    ci.stage.pName = p.entry.c_str();
    ci.layout = p.layout;
    if (vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe) != VK_SUCCESS) {
        vkDestroyPipelineLayout(d.dev, p.layout, nullptr);
        if (p.pool) vkDestroyDescriptorPool(d.dev, p.pool, nullptr);
        if (p.dsl) vkDestroyDescriptorSetLayout(d.dev, p.dsl, nullptr);
        vkDestroyShaderModule(d.dev, p.mod, nullptr);
        why = "vkCreateComputePipelines failed (the SPIR-V entry point or its layout)";
        return fail(kErrorInvalidValue);
    }
    if (spec.shared_mem > d.props.limits.maxComputeSharedMemorySize) {
        // Vulkan's only workgroup-memory limit is per-pipeline: VkPhysicalDeviceLimits has no per-workgroup
        // maxComputeLocalSize field (that name is CUDA's cudaDevAttrMaxSharedMemoryPerBlockOptin).
        note("shared", "the kernel %s asks for %llu B of workgroup-local memory; %s allows %llu B in one pipeline "
                       "(VkPhysicalDeviceLimits::maxComputeSharedMemorySize) and %u invocations per workgroup "
                       "(maxComputeWorkGroupInvocations)\n", p.name.c_str(), (unsigned long long) spec.shared_mem,
             d.props.deviceName, (unsigned long long) d.props.limits.maxComputeSharedMemorySize,
             d.props.limits.maxComputeWorkGroupInvocations);
    }
    const auto slot = d.pipes.emplace(p.key, std::move(p));
    if (slot.first->second.name == "noop") d.noop_pipe = &slot.first->second;
    return kSuccess;
}

VkDescriptorSet chain_set(Chain& c, Pipeline& p, Device& d) {
    auto it = c.sets.find(p.key);
    if (it != c.sets.end()) return it->second;
    if (p.pool == VK_NULL_HANDLE || p.dsl == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = p.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult r = vkAllocateDescriptorSets(d.dev, &ai, &set);
    if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
        vkResetDescriptorPool(d.dev, p.pool, 0);
        r = vkAllocateDescriptorSets(d.dev, &ai, &set);
    }
    if (r != VK_SUCCESS) { note("descsets", "vkAllocateDescriptorSets for %s said %s\n", p.name.c_str(), vk_name(r)); return VK_NULL_HANDLE; }
    c.sets[p.key] = set;
    return set;
}

// Record one dispatch.  A SPIR-V entry point takes its arguments as ONE parameter block, so it goes in as push
// constants when it fits (maxPushConstantSize is 64 B on both Arcs, P1) and as a uniform binding otherwise; each
// uint64 the caller declares a device address becomes one storage binding over that address's own span, with the
// offset taken from the span (which may be past 4 GiB) and the range clamped by bind_span (which may not be).
// `x` is the graph exec being re-encoded: its buffer and its window block are then the target, which is how a replay
// re-uploads the parameter bytes instead of reusing the constants baked in at capture.
Error record_dispatch(Device& d, StreamImpl* s, Pipeline& p, const LaunchConfig& cfg, const void* args, size_t args_bytes,
                      GraphExecImpl* x = nullptr) {
    VkCommandBuffer cb = x != nullptr ? x->cb : record_cb(s);
    if (cb == VK_NULL_HANDLE) return fail(kErrorInvalidValue);

    // Where this dispatch's parameter bytes go: the chain slot's block for a live launch, the graph's block while
    // capturing, the exec's window block for every replay.
    Alloc** pa = nullptr;
    uint64_t* pu = nullptr;
    Chain* ch = s->open;
    if (x != nullptr) { pa = &x->window_alloc; pu = &x->window_used; }
    else if (s->rec != nullptr) { pa = &s->rec->param_alloc; pu = &s->rec->param_used; }
    else if (ch != nullptr) { pa = &ch->param_alloc; pu = &ch->param_used; }

    if (p.push_params && p.param_bytes != 0) {
        uint8_t block[256];
        std::memset(block, 0, sizeof block);
        if (args && args_bytes) std::memcpy(block, args, std::min<size_t>(args_bytes, sizeof block));
        vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p.param_bytes, block);
    } else if (p.param_bytes != 0 && pa != nullptr && pu != nullptr) {
        if (*pa == nullptr) {
            bool oom = false;
            *pa = sub_alloc(d, 64 * kKiB, kHostWant, d.align_uniform, &oom);
            if (*pa == nullptr) { note("params", "no uniform parameter block for a dispatch"); return fail(kErrorMemoryAllocation); }
            *pu = 0;
        }
        Span& sp = (*pa)->spans[0];
        uint8_t* cpu = (uint8_t*) sp.block->map + sp.block_off;
        const uint64_t off = round_up(*pu, 256);
        if (off + p.param_bytes > sp.bytes) { note("params", "the parameter block for this chain / graph is full"); return fail(kErrorMemoryAllocation); }
        if (args && args_bytes) std::memcpy(cpu + off, args, std::min<size_t>(args_bytes, (size_t) p.param_bytes));
        *pu = off + p.param_bytes;

        std::map<std::string, VkDescriptorSet>* sets = nullptr;
        if (x != nullptr) sets = &x->sets;
        else if (ch != nullptr) sets = &ch->sets;
        VkDescriptorSet set = VK_NULL_HANDLE;
        {   // the descriptor set for this (pipeline, target) pair, allocated from the pipeline's own pool
            auto it = sets->find(p.key);
            if (it != sets->end()) set = it->second;
            else if (p.pool != VK_NULL_HANDLE && p.dsl != VK_NULL_HANDLE) {
                VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                ai.descriptorPool = p.pool;
                ai.descriptorSetCount = 1;
                ai.pSetLayouts = &p.dsl;
                VkResult r = vkAllocateDescriptorSets(d.dev, &ai, &set);
                if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
                    vkResetDescriptorPool(d.dev, p.pool, 0);
                    sets->clear();
                    r = vkAllocateDescriptorSets(d.dev, &ai, &set);
                }
                if (r != VK_SUCCESS) { note("descsets", "vkAllocateDescriptorSets for %s said %s\n", p.name.c_str(), vk_name(r)); return fail(kErrorInvalidValue); }
                (*sets)[p.key] = set;
            }
        }
        if (set == VK_NULL_HANDLE) return fail(kErrorInvalidValue);
        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes;
        VkDescriptorBufferInfo ubi;
        bind_span(d, sp.block->buf, sp.block_off + off, p.param_bytes, ubi);
        infos.push_back(ubi);
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = p.param_binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &infos.back();
        writes.push_back(w);
        const uint64_t* words = (const uint64_t*) args;
        for (uint32_t i = 0; i < p.storage_count; ++i) {
            const uint64_t addr = (args && (size_t) (i * 8) < args_bytes) ? words[i] : 0;
            uint64_t aoff = 0;
            bool is_host = false;
            Alloc* a = find_alloc(d, (const void*) (uintptr_t) addr, &aoff, &is_host);
            const Span* sp2 = a != nullptr ? span_at(a, aoff) : nullptr;
            VkDescriptorBufferInfo bi{VK_NULL_HANDLE, 0, 1};
            if (sp2 != nullptr) {
                const uint64_t inside = addr - sp2->addr;                 // the OFFSET may be past 4 GiB ...
                bi.range = std::min<uint64_t>(sp2->bytes - inside, block_limit(d));   // ... the RANGE may not
                bind_span(d, sp2->block->buf, sp2->block_off + inside, bi.range, bi); // ... and both are checked here
            } else {
                note("param-addr", "the kernel %s was given the device address %llu, which is not an allocation this "
                                   "runtime handed out; its storage binding is left empty\n", p.name.c_str(),
                     (unsigned long long) addr);
            }
            infos.push_back(bi);
            VkWriteDescriptorSet w2{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w2.dstSet = set;
            w2.dstBinding = p.param_binding + (p.push_params ? 0 : 1) + i;
            w2.descriptorCount = 1;
            w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w2.pBufferInfo = &infos.back();
            writes.push_back(w2);
        }
        vkUpdateDescriptorSets(d.dev, (uint32_t) writes.size(), writes.data(), 0, nullptr);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    } else if (p.dsl != VK_NULL_HANDLE && ch != nullptr) {
        VkDescriptorSet set = chain_set(*ch, p, d);
        if (set != VK_NULL_HANDLE) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    }

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdDispatch(cb, cfg.grid.x, cfg.grid.y, cfg.grid.z);      // CUDA's <<<grid, block>>>: grid IS the workgroup count
    if (cfg.block.x != p.local[0] || cfg.block.y != p.local[1] || cfg.block.z != p.local[2])
        note("workgroup", "the launch of %s asked for a workgroup of %ux%ux%u but its SPIR-V entry point declares "
                          "%ux%ux%u; the pipeline's own size is what runs\n", p.name.c_str(), cfg.block.x, cfg.block.y,
             cfg.block.z, p.local[0], p.local[1], p.local[2]);
    s->dispatches++;
    s->recorded++;
    if (s->rec == nullptr && s->recorded >= kFlushEvery) nudge_submit(s);   // a window-sized buffer, not one per launch
    return kSuccess;
}

// ------------------------------------------------------------------ the graph node list and its re-encoding

Error record_node(Device& d, StreamImpl* s, GraphExecImpl* x, const NodeImpl& n);

// vkResetCommandBuffer + re-record the node list into the SAME VkCommandBuffer.  A submitted command buffer's contents
// are consumed, so replaying a graph means re-encoding it into the one buffer the capture opened - no new buffer per
// window, and no cudaGraphExecUpdate, which Vulkan has no equivalent of.
Error reencode_graph(Device& d, GraphExecImpl* x) {
    x->window_used = 0;                        // each replay writes its own parameter bytes into the window block
    x->window_bytes = x->window_alloc != nullptr ? x->window_alloc->bytes : 0;
    if (vkResetCommandBuffer(x->cb, 0) != VK_SUCCESS) { note("graph-reset", "vkResetCommandBuffer failed on a captured graph"); return fail(kErrorInvalidValue); }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(x->cb, &bi) != VK_SUCCESS) { note("graph-begin", "vkBeginCommandBuffer failed on a captured graph"); return fail(kErrorInvalidValue); }
    StreamImpl* s = stream_of(d, nullptr);
    for (const NodeImpl* n : x->graph->nodes) {
        const Error e = record_node(d, s, x, *n);
        if (e != kSuccess) return e;
    }
    if (vkEndCommandBuffer(x->cb) != VK_SUCCESS) { note("graph-end", "vkEndCommandBuffer failed on a captured graph"); return fail(kErrorInvalidValue); }
    return kSuccess;
}

Error record_node(Device& d, StreamImpl* s, GraphExecImpl* x, const NodeImpl& n) {
    VkCommandBuffer cb = x != nullptr ? x->cb : record_cb(s);
    if (cb == VK_NULL_HANDLE) return fail(kErrorInvalidValue);
    switch (n.type) {
        case (int) cudaGraphNodeTypeKernel: {
            Pipeline* p = find_pipeline(d, n.name.c_str());
            if (p == nullptr) {
                note("graph-kernel", "the captured graph contains the entry point \"%s\", which is not registered on %s "
                                     "(strata::gpu::register_pipeline is where a kernel registers its SPIR-V)\n",
                     n.name.c_str(), d.props.deviceName);
                return fail(kErrorUnsupported);
            }
            LaunchConfig cfg;
            cfg.grid = n.grid; cfg.block = n.block; cfg.shared_mem = n.shared;
            return record_dispatch(d, s, *p, cfg, n.params.empty() ? nullptr : n.params.data(), n.params.size(), x);
        }
        case (int) cudaGraphNodeTypeMemcpy:
            return copy_region(d, s, n.dst, n.src, n.bytes, kCopyDefault, false, cb);
        case (int) cudaGraphNodeTypeMemset:
            return fill_region(d, s, n.dst, n.value, n.bytes, false, cb);
        default:
            return kSuccess;   // a host node or an event node is not a VkCommand: the launch resolves those
    }
}

// ------------------------------------------------------------------ streams and devices: construction

StreamImpl* make_stream(Device& d, uint32_t qidx, unsigned flags) {
    std::unique_ptr<StreamImpl> s(new StreamImpl);
    s->d = &d;
    s->qidx = qidx;
    s->q = d.queues[qidx];
    s->prio_cuda = flags;

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = d.qfam;
    if (vkCreateCommandPool(d.dev, &pi, nullptr, &s->pool) != VK_SUCCESS) {
        note("pool", "vkCreateCommandPool failed on %s", d.props.deviceName);
        return nullptr;
    }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = s->pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kChainSlots;
    VkCommandBuffer cbs[kChainSlots] = {};
    if (vkAllocateCommandBuffers(d.dev, &ai, cbs) != VK_SUCCESS) {
        note("pool", "vkAllocateCommandBuffers failed on %s", d.props.deviceName);
        vkDestroyCommandPool(d.dev, s->pool, nullptr);
        s->pool = VK_NULL_HANDLE;
        return nullptr;
    }
    for (uint32_t i = 0; i < kChainSlots; ++i) {
        s->chain[i].cb = cbs[i];
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(d.dev, &fi, nullptr, &s->chain[i].fence);
    }
    // A stream's completion is a value on a timeline semaphore, not an event object: CUDA's "has this stream's work
    // finished" is vkWaitSemaphores / vkGetSemaphoreCounterValue against that value.
    VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;                  // core 1.2; Phase 0 P1: VK_KHR_timeline_semaphore present
    ti.initialValue = 0;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    si.pNext = &ti;
    const VkResult sr = vkCreateSemaphore(d.dev, &si, nullptr, &s->tl);
    if (sr != VK_SUCCESS) {
        note("sem", "vkCreateSemaphore(VK_SEMAPHORE_TYPE_TIMELINE) on %s said %s (open_device enables the "
                    "timelineSemaphore feature, so this is the driver refusing the semaphore itself)\n",
             d.props.deviceName, vk_name(sr));
        vkDestroyCommandPool(d.dev, s->pool, nullptr);
        s->pool = VK_NULL_HANDLE;
        return nullptr;
    }
    d.streams.push_back(std::move(s));
    return d.streams.back().get();
}

void destroy_stream(Device& d, StreamImpl* s) {
    if (s == nullptr) return;
    nudge_submit(s);
    for (Chain& c : s->chain) {
        if (c.live) vkWaitForFences(d.dev, 1, &c.fence, VK_TRUE, kFenceWaitNs);
        if (c.param_alloc != nullptr) destroy_alloc(d, c.param_alloc);
        if (c.fence != VK_NULL_HANDLE) vkDestroyFence(d.dev, c.fence, nullptr);
    }
    if (s->tl != VK_NULL_HANDLE) vkDestroySemaphore(d.dev, s->tl, nullptr);
    if (s->pool != VK_NULL_HANDLE) vkDestroyCommandPool(d.dev, s->pool, nullptr);
    auto it = std::find_if(d.streams.begin(), d.streams.end(),
                           [&](const std::unique_ptr<StreamImpl>& x) { return x.get() == s; });
    if (it != d.streams.end()) d.streams.erase(it);
    if (d.legacy == s) d.legacy = nullptr;
    if (d.staging_stream == s) { d.staging_stream = nullptr; d.staging_pending = false; }
}

void read_limits(Device& d, VkInstance inst) {
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &d.drv;
    VkBaseOutStructure* tail = (VkBaseOutStructure*) &d.m3;   // Vulkan's chaining idiom: walk the list as base structs
    d.drv.pNext = &d.idp;
    d.idp.pNext = &d.m3;
    if (d.m4_enabled) {                                   // only when VK_KHR_maintenance4 was enabled on this device
        d.m3.pNext = (VkBaseOutStructure*) &d.m4; tail = (VkBaseOutStructure*) &d.m4; d.has_m4 = true;
    }
    tail->pNext = (VkBaseOutStructure*) &d.hostp;
    d.hostp.pNext = &d.subgroup;
    vkGetPhysicalDeviceProperties2(d.phys, &p2);
    // Read the queried values only AFTER the query: VkPhysicalDeviceExternalMemoryHostPropertiesEXT is filled by
    // vkGetPhysicalDeviceProperties2, and a limit checked before it would always read zero.
    d.has_hostp = d.hostp.minImportedHostPointerAlignment != 0;

    VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    if (ext_available(d.phys, "VK_EXT_memory_budget")) {            // the budget struct, chained onto the properties2 query
        mp2.pNext = &d.budget;
        d.has_memory_budget_ext = true;
    }
    vkGetPhysicalDeviceMemoryProperties2(d.phys, &mp2);
    d.mem = mp2.memoryProperties;
    d.has_budget = d.has_memory_budget_ext;

    d.has_idp = d.idp.deviceLUIDValid != 0;
    // STRATA_VULKAN_HOST_IMPORT=0 turns the in-place host mapping off for this card: every host pointer then travels
    // through the mapped staging buffer instead of being imported.  It exists because releasing an import is the one
    // thing in this runtime this driver has intermittently answered with VK_ERROR_DEVICE_LOST (see the note in
    // free_host); a caller that hits it has a working fallback that costs a copy instead of a run.
    d.has_host_import = ext_available(d.phys, "VK_EXT_external_memory_host") &&
                        env_on("STRATA_VULKAN_HOST_IMPORT", true);
    d.block_limit = block_limit(d);
    d.align_storage = std::max<uint64_t>(1, d.props.limits.minStorageBufferOffsetAlignment);   // 64 on both Arcs
    d.align_uniform = std::max<uint64_t>(d.align_storage, std::max<uint64_t>(1, d.props.limits.minUniformBufferOffsetAlignment));
    d.heap_local = device_local_heap(d);
    d.heap_local_bytes = d.mem.memoryHeaps[d.heap_local].size;

    d.get_address = (PFN_vkGetBufferDeviceAddress) vkGetDeviceProcAddr(d.dev, "vkGetBufferDeviceAddress");
    if (d.get_address == nullptr)
        d.get_address = (PFN_vkGetBufferDeviceAddress) vkGetDeviceProcAddr(d.dev, "vkGetBufferDeviceAddressKHR");
    d.get_host_memory = (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetInstanceProcAddr(inst, "vkGetMemoryHostPointerPropertiesEXT");
}

bool open_device(Ctx& c, VkPhysicalDevice pd, VkInstance inst, const VkPhysicalDeviceProperties& props,
                 const VkPhysicalDeviceDriverProperties& drv, std::string& why) {
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn ? qn : 1);
    if (qn) vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs.data());
    int fam = -1;
    for (uint32_t i = 0; i < qn; ++i) {                             // a compute queue is what a kernel needs
        if (!(qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
        if (fam < 0 || qs[i].queueCount > qs[(uint32_t) fam].queueCount) fam = (int) i;
    }
    if (fam < 0) { why = "no queue family offers VK_QUEUE_COMPUTE_BIT"; return false; }

    // What this port opens each device with.  The instance asks for API 1.3 (build_ctx), so a card that supports 1.4 is
    // still opened as a 1.3 device: what 1.4 adds (maxBufferSize) and what 1.2 adds (timeline semaphores, buffer
    // device address) are then reachable only through the extension of the same name, named here.  A feature whose
    // extension is not enabled is a validation error, so each feature struct below is chained only when its extension
    // is in this list.
    const bool use_host_import = ext_available(pd, "VK_EXT_external_memory_host");
    const bool use_budget = ext_available(pd, "VK_EXT_memory_budget");
    const bool use_m4 = ext_available(pd, "VK_KHR_maintenance4");
    const bool use_tl = props.apiVersion < VK_MAKE_VERSION(1, 2, 0) && ext_available(pd, "VK_KHR_timeline_semaphore");
    const bool use_bda = props.apiVersion < VK_MAKE_VERSION(1, 2, 0) && ext_available(pd, "VK_KHR_buffer_device_address");
    std::vector<const char*> use;
    if (use_host_import) use.push_back("VK_EXT_external_memory_host");
    if (use_budget) use.push_back("VK_EXT_memory_budget");
    if (use_m4) use.push_back("VK_KHR_maintenance4");
    if (use_tl) use.push_back("VK_KHR_timeline_semaphore");
    if (use_bda) use.push_back("VK_KHR_buffer_device_address");

    // As many queues as the card hands out, up to kStreamQueues: one VkQueue per stream.  Priorities descend so the
    // first queue is the one the engine's critical path (the verify stream) gets.
    const uint32_t qcount = std::min<uint32_t>(kStreamQueues, std::max<uint32_t>(1, qs[(uint32_t) fam].queueCount));
    std::vector<float> prio(qcount);
    for (uint32_t i = 0; i < qcount; ++i) prio[i] = 1.0f - 0.5f * (float) i / (float) std::max<uint32_t>(1, qcount - 1);

    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feats.features.shaderInt64 = VK_TRUE;                 // device addresses in a kernel's parameter block are 64-bit
    VkPhysicalDeviceScalarBlockLayoutFeatures sbl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
    sbl.scalarBlockLayout = VK_TRUE;                      // what vulkan/shaders and the probe's shaders are written with
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bda.bufferDeviceAddress = VK_TRUE;                    // vkGetBufferDeviceAddress, which P2 measured
    // Timeline semaphores are CORE in Vulkan 1.2, but "core" means the entry points and the structs: the ability to
    // create one is a feature, and asking for VK_SEMAPHORE_TYPE_TIMELINE without enabling it is what the validation
    // layer said in build-vulkan/logs/s3-smoke-01.log:3 on the first smoke run of this file (the control probe does it on purpose):
    //   VUID-VkSemaphoreTypeCreateInfo-timelineSemaphore-03252: vkCreateSemaphore(): pCreateInfo->semaphoreType is
    //   VK_SEMAPHORE_TYPE_TIMELINE, but timelineSemaphore feature was not enabled.
    // So the feature is named here, on the same VkDeviceCreateInfo the rest of the features ride on.
    VkPhysicalDeviceTimelineSemaphoreFeatures tsf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    tsf.timelineSemaphore = VK_TRUE;                      // every stream's completion and every event is one of these
    VkPhysicalDeviceMaintenance4Features m4f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES};
    feats.pNext = &sbl;
    sbl.pNext = &bda;
    bda.pNext = &tsf;
    if (use_m4) { m4f.maintenance4 = VK_TRUE; tsf.pNext = &m4f; }   // only with the extension it belongs to

    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = (uint32_t) fam;
    qi.queueCount = qcount;
    qi.pQueuePriorities = prio.data();
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.pNext = &feats;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = (uint32_t) use.size();
    di.ppEnabledExtensionNames = use.empty() ? nullptr : use.data();

    std::unique_ptr<Device> dp(new Device);
    Device& d = *dp;
    d.phys = pd;
    d.props = props;
    d.drv = drv;
    d.qfam = (uint32_t) fam;
    d.qfam_queues = qs[(uint32_t) fam].queueCount;
    d.prio = prio;
    d.m4_enabled = use_m4;
    const VkResult r = vkCreateDevice(pd, &di, nullptr, &d.dev);
    if (r != VK_SUCCESS) { why = std::string("vkCreateDevice said ") + vk_name(r); return false; }
    d.queues.resize(qcount);
    for (uint32_t i = 0; i < qcount; ++i) vkGetDeviceQueue(d.dev, d.qfam, i, &d.queues[i]);

    read_limits(d, inst);
    d.legacy = make_stream(d, 0, 0);
    if (d.legacy == nullptr) { why = "the default stream could not be created"; vkDestroyDevice(d.dev, nullptr); return false; }
    for (uint32_t i = 1; i < qcount; ++i) make_stream(d, i, 0);
    d.ordinal = (uint32_t) c.devs.size();
    c.max_import_align = std::max<uint64_t>(c.max_import_align, std::max<uint64_t>(1, d.hostp.minImportedHostPointerAlignment));

#ifdef STRATA_VULKAN_HAS_NOOP_SPV
    // The one shader this port ships in Phase 1, built from vulkan/shaders/noop.comp by vulkan/CMakeLists.txt.  Its
    // body is empty: no loop, no counter, no host-written flag to poll - Phase 0 P5 measured that this driver never
    // makes a host write visible to a dispatch already running, so this runtime signals with submits and timelines.
    // Building the pipeline here is the proof the SPIR-V -> VkShaderModule -> VkPipeline path works on this card; the
    // engine dispatches nothing in Phase 1 (its kernel entry points are still the generated stubs).
    static const std::vector<uint32_t> noop_words = [] {
        std::vector<uint32_t> w((noop_spv_len + 3) / 4, 0u);          // vkCreateShaderModule wants whole, aligned words
        std::memcpy(w.data(), noop_spv, noop_spv_len);
        return w;
    }();
    PipelineSpec spec;
    spec.name = "noop";
    spec.entry = "main";
    spec.spv = noop_words.data();
    spec.spv_words = (size_t) noop_words.size();
    spec.workgroup = Dim3{1, 1, 1};
    spec.param_bytes = 8;                     // the shader's two push-constant uints
    spec.storage_bindings = 1;                // its one storage binding, binding 0
    std::string pwhy;
    if (build_pipeline(d, spec, pwhy) != kSuccess) note("noop-pipe", "%s: %s\n", d.props.deviceName, pwhy.c_str());
#endif

    c.devs.push_back(std::move(dp));
    return true;
}

void build_ctx(Ctx& c) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (c.tried) return;
    c.tried = true;

    std::vector<const char*> iext = {"VK_KHR_get_physical_device_properties2", "VK_KHR_portability_enumeration"};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "strata";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 44);
    app.pEngineName = "strata vulkan";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 44);
    app.apiVersion = VK_API_VERSION_1_3;              // the version the plan's mapping table needs (timeline semaphores)
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t) iext.size();
    ici.ppEnabledExtensionNames = iext.data();
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    VkResult r = vkCreateInstance(&ici, nullptr, &c.inst);
    if (r != VK_SUCCESS) { c.why = std::string("vkCreateInstance said ") + vk_name(r); return; }
    // The instance and every VkDevice live for the whole process, as the Metal runtime's context does: the engine
    // keeps raw pointers to buffers across phases and never tears the device down itself.

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
    if (n == 0) { c.why = "vkEnumeratePhysicalDevices reported no device"; return; }
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(c.inst, &n, phys.data());

    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys[i], &props);
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &drv;
        vkGetPhysicalDeviceProperties2(phys[i], &p2);

        std::string why;
        if (!choose_physical(props, drv, why)) { c.skipped.push_back(std::string(props.deviceName) + ": " + why); continue; }
        if (!open_device(c, phys[i], c.inst, props, drv, why)) { c.skipped.push_back(std::string(props.deviceName) + ": " + why); continue; }
    }
    if (c.devs.empty()) {
        c.why = c.skipped.empty() ? "no physical device was enumerated" : "every enumerated device was skipped: " + c.skipped[0];
        return;
    }
    c.ok = true;
    selfcheck(c);
}

Ctx& ctx() {
    static Ctx c = [] { Ctx x; build_ctx(x); return x; }();
    return c;
}

// The guarded, one-shot dump: what the selection kept and dropped, the limits the rules below are read from, and the
// heap budgets cudaMemGetInfo is computed from.  Only a run that opens the runtime reaches it.
void selfcheck(Ctx& c) {
    const bool on = env_on("STRATA_VULKAN_SELFCHECK", true);
    if (!on) return;
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    std::fprintf(stderr, "strata vulkan self-check (one-shot; STRATA_VULKAN_SELFCHECK=0 silences this)\n");
    std::fprintf(stderr, "  Vulkan API requested by this build: %u.%u\n",
                 VK_VERSION_MAJOR(VK_API_VERSION_1_3), VK_VERSION_MINOR(VK_API_VERSION_1_3));
    for (const auto& d : c.devs) {
        const Device& x = *d;
        std::fprintf(stderr, "  device %u: %s  api %u.%u.%u  vendor %04x device %04x  driver \"%s\" id %u \"%s\"\n",
                     x.ordinal, x.props.deviceName, VK_VERSION_MAJOR(x.props.apiVersion), VK_VERSION_MINOR(x.props.apiVersion),
                     VK_VERSION_PATCH(x.props.apiVersion), x.props.vendorID, x.props.deviceID, x.drv.driverName,
                     (unsigned) x.drv.driverID, x.drv.driverInfo);
        std::fprintf(stderr, "    limits: maxStorageBufferRange %llu MiB  maxBufferSize %llu MiB  maxMemoryAllocationSize "
                             "%llu MiB  -> every block and every single storage binding is clamped to %llu MiB\n",
                     (unsigned long long) (x.props.limits.maxStorageBufferRange / kMiB),
                     x.has_m4 ? (unsigned long long) (x.m4.maxBufferSize / kMiB) : 0ull,
                     (unsigned long long) (x.m3.maxMemoryAllocationSize / kMiB),
                     (unsigned long long) (x.block_limit / kMiB));
        std::fprintf(stderr, "    alignment: minStorageBufferOffsetAlignment %llu (every sub-allocation offset is "
                             "rounded up to it)  minUniformBufferOffsetAlignment %llu  bufferImageGranularity %llu\n",
                     (unsigned long long) x.align_storage, (unsigned long long) x.props.limits.minUniformBufferOffsetAlignment,
                     (unsigned long long) x.props.limits.bufferImageGranularity);
        std::fprintf(stderr, "    compute: queues %u of family %u (the card offers %u)  maxComputeWorkGroupCount %u "
                             "maxComputeWorkGroupSize %u maxComputeWorkGroupInvocations %u  "
                             "maxComputeSharedMemorySize %llu B  subgroupSize %u  maxPushConstantsSize %u  "
                             "maxMemoryAllocationCount %u\n",
                     (unsigned) x.queues.size(), x.qfam, x.qfam_queues,
                     x.props.limits.maxComputeWorkGroupCount[0], x.props.limits.maxComputeWorkGroupSize[0],
                     x.props.limits.maxComputeWorkGroupInvocations,
                     (unsigned long long) x.props.limits.maxComputeSharedMemorySize,
                     x.subgroup.subgroupSize, x.props.limits.maxPushConstantsSize, x.props.limits.maxMemoryAllocationCount);
        for (uint32_t h = 0; h < x.mem.memoryHeapCount; ++h) {
            std::fprintf(stderr, "    heap %u: %llu MiB %s  budget %llu MiB  usage now %llu MiB%s\n", h,
                         (unsigned long long) (x.mem.memoryHeaps[h].size / kMiB),
                         (x.mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "device-local" : "host heap",
                         x.has_budget ? (unsigned long long) (x.budget.heapBudget[h] / kMiB) : 0ull,
                         x.has_budget ? (unsigned long long) (x.budget.heapUsage[h] / kMiB) : 0ull,
                         h == x.heap_local ? "  <- cudaMemGetInfo's heap" : "");
        }
        std::fprintf(stderr, "    this process's own device allocations right now: %llu MiB (cudaMemGetInfo subtracts "
                             "the larger of this and the driver's own usage counter)\n",
                     (unsigned long long) (x.own_device_bytes / kMiB));
        std::fprintf(stderr, "    VK_EXT_external_memory_host %s (minImportedHostPointerAlignment %llu KiB)  "
                             "VK_EXT_memory_budget %s  buffer device address %s  LUID %s\n",
                     x.has_host_import ? "present" : "absent",
                     (unsigned long long) (x.hostp.minImportedHostPointerAlignment / kKiB),
                     x.has_memory_budget_ext ? "chained onto the properties query" : "absent",
                     x.get_address ? "yes" : "no", x.has_idp ? "yes" : "no");
        std::fprintf(stderr, "    noop SPIR-V pipeline: %s (built from vulkan/shaders/noop.comp; nothing in Phase 1 "
                             "dispatches a kernel)\n", x.noop_pipe ? "created" : "not available");
    }
    for (const auto& s : c.skipped) std::fprintf(stderr, "  skipped: %s\n", s.c_str());
    std::fprintf(stderr, "  waits: a host->GPU signal is a vkQueueSubmit (nudge_submit), a GPU->host signal is a "
                         "timeline semaphore value (vkWaitSemaphores).  No shader in this port polls a host-written "
                         "flag (Phase 0 P5).\n");
    std::fflush(stderr);
}

}  // namespace

// ============================================================================================
// The CUDA-facing face: every name vulkan_compat/cuda_runtime.h forwards here, and what Vulkan
// answers it with.  Nothing below invents a number: where this port has no Vulkan answer (a clock
// rate, a compute-unit count) it says so by returning an error the caller already handles.
// ============================================================================================

Caps caps() {
    Caps c;
    c.unified_memory = false;                    // discrete Arcs: device-local buffers, explicit copies
    c.host_mapped_waits = false;                 // P5: no host-written flag a running dispatch can see; waits are submits
    return c;
}

// The CPU view of an allocation: the mapped address of a host-visible block, or the host allocation itself for an
// import.  nullptr - with the reason named once - for a device-only memory type, which is what
// vulkan/src/core/device_vk.cpp:131-135 reports rather than writing through.
void* host_view(const void* device_pointer) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("host_view");
    if (d == nullptr) return nullptr;
    uint64_t off = 0;
    bool is_host = false;
    Alloc* a = find_alloc(*d, device_pointer, &off, &is_host);
    if (a == nullptr) {
        note("host-view", "host_view(%p): not an allocation this runtime handed out, so there is no CPU view of it\n",
             device_pointer);
        return nullptr;
    }
    const uint64_t base = is_host ? (uint64_t) (uintptr_t) a->host : a->addr;
    const Span* s = span_at(a, (uint64_t) (uintptr_t) device_pointer - base);
    if (s == nullptr) return nullptr;
    if (s->block->map == nullptr) {
        note("host-view", "host_view(%p): its block's memory type (%s) is not HOST_VISIBLE, so this card has no CPU "
                          "view of it; a vkCmdCopyBuffer into a HOST_VISIBLE|HOST_COHERENT staging buffer is the way "
                          "to read it (both Arcs do offer a DEV_LOCAL|HOST_VISIBLE|HOST_COHERENT type, and that is "
                          "what this runtime asks for first)\n", device_pointer, s->block->dev->props.deviceName);
        return nullptr;
    }
    return (char*) s->block->map + s->block_off + ((uint64_t) (uintptr_t) device_pointer - s->addr);
}

Error get_last_error() { return g_last_error.exchange(kSuccess, std::memory_order_relaxed); }
Error peek_last_error() { return g_last_error.load(std::memory_order_relaxed); }

const char* error_name(Error e) {
    switch (e) {
        case kSuccess: return "cudaSuccess";
        case kErrorInvalidValue: return "cudaErrorInvalidValue";
        case kErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case kNotReady: return "cudaErrorNotReady";
        case kErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";
        case kErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case kErrorUnsupported: return "cudaErrorNotSupported";
        default: return "cudaErrorUnknown";
    }
}

const char* error_string(Error e) {
    switch (e) {
        case kSuccess: return "no error";
        case kErrorInvalidValue: return "invalid value (a pointer this runtime does not know, or a bad size)";
        case kErrorMemoryAllocation: return "out of memory (the sub-allocator could not open a block within the card's limit)";
        case kNotReady: return "the operation is still pending (the timeline value has not been reached yet)";
        case kErrorPeerAccessAlreadyEnabled: return "peer access is already enabled (Vulkan has no peer API: two VkDevices)";
        case kErrorLaunchFailure: return "a submit failed or the device was lost (VkResult from vkQueueSubmit / vkWaitSemaphores)";
        case kErrorUnsupported: return "not supported on Vulkan (see the named line on stderr for the Vulkan call that will answer it)";
        default: return "unknown error";
    }
}

Error get_device(int* dev) {
    if (dev) *dev = t_current_device;
    return kSuccess;
}

Error set_device(int dev) {
    Ctx& c = ctx();
    if (!c.ok || dev < 0 || (size_t) dev >= c.devs.size()) {
        note("set-device", "cudaSetDevice(%d): this port has %zu device(s) - %s\n", dev, c.devs.size(),
             c.why.empty() ? "the ones the selection kept" : c.why.c_str());
        return fail(kErrorInvalidValue);
    }
    t_current_device = dev;
    return kSuccess;
}

Error device_count(int* n) {
    Ctx& c = ctx();
    if (n) *n = (int) c.devs.size();
    return kSuccess;      // 0 with the reason on stderr is the honest answer when there is nothing
}

Error device_synchronize() {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaDeviceSynchronize");
    if (d == nullptr) return fail(kErrorUnsupported);
    drain_readback(*d);
    for (auto& s : d->streams) {
        nudge_submit(s.get());
        const Error e = wait_timeline(*d, s->tl, s->signaled, "cudaDeviceSynchronize");
        if (e != kSuccess) return e;
    }
    return kSuccess;
}

// cudaMemGetInfo, from VK_EXT_memory_budget.  heapBudget is what the driver lets one application use of that heap
// (15489 MiB of the A770's 16258, 11349 of the B580's 12118 - bench/results/vulkan-caps-2026-10-17.md P1) and
// heapUsage is what is on that heap now, this process and every other one.  The free figure is the budget minus what
// is already committed, and it also charges this runtime's own live device allocations, because the sub-allocator has
// taken those from the same budget and the engine's next cudaMalloc is measured against what is left to hand out.
Error mem_get_info(size_t* free_b, size_t* total_b) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* dp = current_device("cudaMemGetInfo");
    if (dp == nullptr) return fail(kErrorUnsupported);
    Device& d = *dp;
    read_budget(d);                                    // re-read: the driver's usage numbers move as the run goes on
    const uint32_t h = d.heap_local;
    uint64_t total = d.heap_local_bytes;
    if (d.has_budget && d.budget.heapBudget[h] != 0) total = d.budget.heapBudget[h];   // the budget, not the raw heap size
    const uint64_t own = d.own_device_bytes;           // this process's own live device allocations
    const uint64_t usage = d.has_budget ? d.budget.heapUsage[h] : 0;
    const uint64_t used = std::max(own, usage);        // never count the same bytes twice
    const uint64_t available = total > used ? total - used : 0;   // the budget, minus what is already committed
    if (free_b) *free_b = (size_t) available;
    if (total_b) *total_b = (size_t) total;
    return kSuccess;
}

namespace {   // the same anonymous namespace the machinery above lives in: a definition outside it would be a
              // different function, which is what LNK2019 on read_budget said
void read_budget(Device& d) {
    if (!d.has_memory_budget_ext) return;
    VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    mp2.pNext = &d.budget;                             // VkPhysicalDeviceMemoryBudgetPropertiesEXT on the query
    vkGetPhysicalDeviceMemoryProperties2(d.phys, &mp2);
    d.mem = mp2.memoryProperties;
}
}  // namespace

Error get_device_properties(DeviceProp* p, int dev) {
    Ctx& c = ctx();
    if (!c.ok || dev < 0 || (size_t) dev >= c.devs.size()) return fail(kErrorInvalidValue);
    const Device& d = *c.devs[(size_t) dev];
    if (p == nullptr) return fail(kErrorInvalidValue);
    std::memset(p, 0, sizeof *p);
    std::snprintf(p->name, sizeof p->name, "%s", d.props.deviceName);
    p->totalGlobalMem = d.heap_local_bytes;            // the device-local heap: 16258 MiB A770, 12118 MiB B580
    p->major = (int) VK_VERSION_MAJOR(d.props.apiVersion);
    p->minor = (int) VK_VERSION_MINOR(d.props.apiVersion);   // Vulkan has no compute capability: this carries the API version
    p->warpSize = (int) d.subgroup.subgroupSize;       // 32 on both Arcs (P1)
    p->multiProcessorCount = 0;                        // VkPhysicalDeviceLimits has no unit count: see device_sm_count
    p->memoryClockRate = 0;
    p->memoryBusWidth = 0;
    p->l2CacheSize = 0;                        // VkPhysicalDeviceLimits reports no L2 size; 0 means "unknown", not "none"
    p->maxSharedMemoryPerOptin = (int) d.props.limits.maxComputeSharedMemorySize;   // Vulkan's one workgroup-memory
                                                                                     // limit, per pipeline
    if (d.has_idp) {
        std::memcpy(p->luid, d.idp.deviceLUID, sizeof p->luid);   // VK_LUID_SIZE is 8: exactly CUDA's luid[8]
        p->luidDeviceNodeMask = d.idp.deviceNodeMask;
    }
    return kSuccess;
}

Error device_integrated(int* v, int dev) {
    Ctx& c = ctx();
    if (!c.ok || dev < 0 || (size_t) dev >= c.devs.size()) return fail(kErrorInvalidValue);
    if (v) *v = c.devs[(size_t) dev]->props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
    return kSuccess;   // always 0 here: choose_physical dropped the integrated ones before a VkDevice existed
}

Error device_sm_count(int* v, int dev) {
    Ctx& c = ctx();
    if (!c.ok || dev < 0 || (size_t) dev >= c.devs.size()) return fail(kErrorInvalidValue);
    note("sm-count", "cudaDevAttrMultiProcessorCount has no Vulkan equivalent: VkPhysicalDeviceLimits reports no "
                     "compute-unit count.  What the card does report is maxComputeWorkGroupCount %u, "
                     "maxComputeWorkGroupSize %u and subgroupSize %u (strata::plan does not multiply those out into a "
                     "unit count on purpose).\n",
         c.devs[(size_t) dev]->props.limits.maxComputeWorkGroupCount[0],
         c.devs[(size_t) dev]->props.limits.maxComputeWorkGroupSize[0], c.devs[(size_t) dev]->subgroup.subgroupSize);
    if (v) *v = 0;
    return fail(kErrorUnsupported);
}

Error device_clock_khz(int* v, int dev) {
    Ctx& c = ctx();
    if (!c.ok || dev < 0 || (size_t) dev >= c.devs.size()) return fail(kErrorInvalidValue);
    note("clock", "cudaDevAttrClockRate / cudaDevAttrMemoryClockRate / cudaDevAttrGlobalMemoryBusWidth have no Vulkan "
                  "equivalent either: VkPhysicalDeviceLimits reports no clock rate and no bus width, so src/program/"
                  "generate.cpp:3829 keeps its own fallback for its speed estimate.\n");
    if (v) *v = 0;
    return fail(kErrorUnsupported);
}

Error device_attr_int(int* v, int attr, int dev) {
    Ctx& c = ctx();
    if (!c.ok || dev < 0 || (size_t) dev >= c.devs.size()) return fail(kErrorInvalidValue);
    const Device& d = *c.devs[(size_t) dev];
    switch (attr) {
        case 18:  return device_integrated(v, dev);                       // cudaDevAttrIntegrated
        case 16:  return device_sm_count(v, dev);                         // cudaDevAttrMultiProcessorCount
        case 13:  case 36:  case 37: return device_clock_khz(v, dev);     // clock / memory clock / bus width
        case 97:  if (v) *v = (int) d.props.limits.maxComputeSharedMemorySize; return kSuccess;  // ...PerBlockOptin
        case 75:  if (v) *v = (int) VK_VERSION_MAJOR(d.props.apiVersion); return kSuccess; // compute capability: the API version
        case 76:  if (v) *v = (int) VK_VERSION_MINOR(d.props.apiVersion); return kSuccess;
        default:  note("attr", "cudaDeviceGetAttribute(%d) is not answered by this port yet", attr); return fail(kErrorUnsupported);
    }
}

Error runtime_version(int* v) {
    Ctx& c = ctx();
    uint32_t best = 0;
    for (const auto& d : c.devs) best = std::max(best, d->props.apiVersion);
    if (v) *v = best != 0 ? (int) (VK_VERSION_MAJOR(best) * 1000 + VK_VERSION_MINOR(best) * 10) : 12000;
    return kSuccess;
}

Error device_can_access_peer(int* v, int dev, int peer) {
    (void) dev; (void) peer;
    if (v) *v = 0;                    // two VkDevices: Vulkan's answer to CUDA's peer access is a host-side hand-off
    return kSuccess;
}

Error device_enable_peer_access(int peer, unsigned flags) {
    (void) peer; (void) flags;
    note("peer", "cudaDeviceEnablePeerAccess has no Vulkan equivalent: this port has one VkDevice per card and no "
                 "NVLink, so a cross-card hand-off is staged on the host, as the SYCL split does "
                 "(src/core/expert_source.cpp:3424).");
    return fail(kErrorUnsupported);
}

Error device_set_flags(unsigned flags) {
    note("sched", "cudaSetDeviceFlags(%u): spin / yield / blocking-sync have no Vulkan equivalent.  What this port "
                  "does instead is vkWaitSemaphores in %llu s slices (wait_timeline), so a lost device is reported "
                  "instead of hung on.", flags, (unsigned long long) (kWaitSliceNs / 1000000000ull));
    return kSuccess;
}

Error device_map_host() { return kSuccess; }   // HOST_VISIBLE|HOST_COHERENT mapped memory is always available here

// ---- memory -------------------------------------------------------------------------------

Error malloc_device_raw(void** p, size_t bytes) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* dp = current_device("cudaMalloc");
    if (dp == nullptr) return fail(kErrorUnsupported);
    Device& d = *dp;
    bool oom = false;
    Alloc* a = sub_alloc(d, bytes, kDeviceWant, d.align_storage, &oom);
    if (a == nullptr) {
        const uint64_t lim = block_limit(d);
        if (oom) {
            size_t free_b = 0, total_b = 0;
            mem_get_info(&free_b, &total_b);
            note("malloc-oom", "cudaMalloc(%llu MiB) on %s failed: the budget says %llu MiB of %llu MiB is free, and "
                               "one block is clamped to %llu MiB.\n", (unsigned long long) (bytes / kMiB),
                 d.props.deviceName, (unsigned long long) (free_b / kMiB), (unsigned long long) (total_b / kMiB),
                 (unsigned long long) (lim / kMiB));
        }
        return fail(kErrorMemoryAllocation);
    }
    register_alloc(a);
    if (p) *p = (void*) (uintptr_t) a->addr;            // a device address, as CUDA hands one back
    return kSuccess;
}

Error free_device(void* p) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaFree");
    if (d == nullptr) return fail(kErrorUnsupported);
    Alloc* a = find_alloc(*d, p, nullptr, nullptr);
    if (a == nullptr) {
        if (p == nullptr) return kSuccess;              // CUDA: cudaFree(nullptr) is fine
        note("free", "cudaFree(%p): this runtime never handed that pointer out, so there is nothing to give back\n", p);
        return fail(kErrorInvalidValue);
    }
    destroy_alloc(*d, a);
    return kSuccess;
}

// cudaHostAlloc / cudaMallocHost: an aligned host allocation (4 KiB, the import alignment both Arcs report) imported
// into every device this process opened, which is what makes cudaHostGetDevicePointer's alias real.  Mapped is
// therefore not a lie here: the same pages are what the CPU writes and what the GPU reads.
Error host_alloc_raw(void** p, size_t bytes, unsigned flags) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Ctx& c = ctx();
    if (!c.ok) { note("host-alloc", "cudaHostAlloc: no device is open (%s)", c.why.c_str()); return fail(kErrorUnsupported); }
    size_t host_capacity = 0;
    void* q = host_pages_acquire(bytes, c.max_import_align, host_capacity);
    if (q == nullptr) return fail(kErrorMemoryAllocation);
    if (flags & kHostAllocWriteCombined)
        note("wc", "cudaHostAllocWriteCombined: this port cannot ask the Vulkan loader for write-combined host "
                   "memory; the allocation is ordinary cached memory and the import is the same.");
    bool any = false;
    for (auto& dp : c.devs) {
        Device& d = *dp;
        if (!d.has_host_import) continue;
        Block* blk = nullptr;
        const VkResult r = import_block(d, q, bytes, blk);
        if (r != VK_SUCCESS || blk == nullptr) {
            note("host-import", "%s: importing the host allocation %p (%llu B) said %s; that card will reach it "
                                "through the staging buffer instead\n", d.props.deviceName, q,
                 (unsigned long long) bytes, vk_name(r));
            continue;
        }
        std::unique_ptr<Alloc> a(new Alloc);
        a->dev = &d;
        a->host = q;
        a->bytes = bytes;
        a->host_alloc = true;
        a->imported = true;
        a->addr = blk->addr;
        a->spans.push_back(Span{blk, (uint64_t) ((uintptr_t) q - (uintptr_t) blk->host_ptr), blk->addr, bytes});
        d.host_imports[(uint64_t) (uintptr_t) q] = 0;    // placeholder, set below once the Alloc is fixed in place
        d.allocs.push_back(std::move(a));
        Alloc* ap = d.allocs.back().get();
        d.by_host[(uint64_t) (uintptr_t) q] = ap;
        d.by_dev[ap->addr] = ap;
        d.host_imports[(uint64_t) (uintptr_t) q] = ap->addr;
        any = true;
    }
    if (!any) {
        // cudaHostAlloc still has to SUCCEED here.  The engine's host side (the expert arena, the read buffers) is
        // ordinary pinned host memory and does not need a device alias to work; what it loses is the alias, and that
        // is what cudaHostGetDevicePointer refuses for, by name, below.  Failing the allocation instead would turn a
        // missing alias into an allocation failure the engine has no way to distinguish from being out of memory.
        note("host-alloc", "no device in this process has VK_EXT_external_memory_host enabled (it is absent, or "
                           "STRATA_VULKAN_HOST_IMPORT=0), so cudaHostAlloc returns plain host memory: host-side work "
                           "and copies through the mapped staging buffer still work, and cudaHostGetDevicePointer "
                           "refuses rather than handing back the CPU address as if a kernel could read it "
                           "(Phase 0 finding 5).\n");
        if (p) *p = q;
        return kSuccess;
    }
    if (p) *p = q;
    return kSuccess;
}

Error free_host(void* p) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Ctx& c = ctx();
    if (p == nullptr) return kSuccess;
    for (auto& dp : c.devs) {
        Device& d = *dp;
        auto it = d.by_host.find((uint64_t) (uintptr_t) p);
        if (it == d.by_host.end() || it->second == nullptr || !it->second->imported) continue;
        Alloc* a = it->second;
        for (Span& s : a->spans) {
            if (s.block->buf != VK_NULL_HANDLE) vkDestroyBuffer(d.dev, s.block->buf, nullptr);
            if (s.block->mem != VK_NULL_HANDLE) vkFreeMemory(d.dev, s.block->mem, nullptr);
            s.block->buf = VK_NULL_HANDLE; s.block->mem = VK_NULL_HANDLE; s.block->map = nullptr; s.block->live = 0;
        }
        d.host_imports.erase((uint64_t) (uintptr_t) p);
        destroy_alloc(d, a);
    }
    // No drain here, and that is a measured decision rather than an oversight: adding one (nudge + wait_timeline on
    // every stream of every card that had these pages mapped, before the pages go back) changed nothing - 4 of the 8
    // runs that followed still lost the A770, against 2 of 6 without it.  What is left is the order the spec asks
    // for: the VkBuffer and its imported VkDeviceMemory are released first.  The pages themselves are not returned to
    // the C allocator at all - see the note on host_pages_acquire, which is the change that stopped the losses.
    host_pages_release(p);                    // kept resident, reused by the next cudaHostAlloc
    return kSuccess;
}

Error host_device_pointer_raw(void** dev, void* host) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaHostGetDevicePointer");
    if (d == nullptr) return fail(kErrorUnsupported);
    auto it = d->host_imports.find((uint64_t) (uintptr_t) host);
    if (it == d->host_imports.end()) {
        // CUDA answers this for any pointer it can map.  This port answers it only for a host allocation it imported,
        // because handing back a CPU address as a device pointer is exactly the bug that makes a kernel read garbage.
        note("host-ptr", "cudaHostGetDevicePointer(%p): this port only aliases a host allocation it imported with "
                         "VK_EXT_external_memory_host.  cudaHostAlloc(Mapped) or cudaHostRegister first; passing the "
                         "CPU address to a kernel is not an option on a discrete card.\n", host);
        return fail(kErrorUnsupported);
    }
    if (dev) *dev = (void*) (uintptr_t) it->second;
    return kSuccess;
}

Error host_register(void* p, size_t bytes, unsigned flags) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Ctx& c = ctx();
    if (!c.ok) return fail(kErrorUnsupported);
    if (p == nullptr || bytes == 0) return fail(kErrorInvalidValue);
    if (flags & kHostRegisterReadOnly)
        note("ro", "cudaHostRegisterReadOnly: Vulkan's host-pointer import has no read-only flavour, so the mapping "
                   "is readable and writable by the GPU.  Nothing in this port writes through a registered arena.");
    int done = 0;
    for (auto& dp : c.devs) {
        Device& d = *dp;
        if (!d.has_host_import) continue;
        if (d.host_imports.count((uint64_t) (uintptr_t) p)) { ++done; continue; }   // already imported (Portable)
        Block* blk = nullptr;
        const VkResult r = import_block(d, p, bytes, blk);
        if (r != VK_SUCCESS || blk == nullptr) {
            note("host-reg", "%s: cudaHostRegister(%p, %llu B) could not be imported (%s).  The card's "
                             "minImportedHostPointerAlignment is %llu KiB, so the pointer must be aligned to that; "
                             "expert_source.cpp's paced registration already checks alignment before asking.\n",
                 d.props.deviceName, p, (unsigned long long) bytes, vk_name(r),
                 (unsigned long long) (d.hostp.minImportedHostPointerAlignment / kKiB));
            continue;
        }
        std::unique_ptr<Alloc> a(new Alloc);
        a->dev = &d;
        a->host = p;
        a->bytes = bytes;
        a->host_alloc = true;
        a->imported = true;
        a->addr = blk->addr;
        a->spans.push_back(Span{blk, (uint64_t) ((uintptr_t) p - (uintptr_t) blk->host_ptr), blk->addr, bytes});
        d.allocs.push_back(std::move(a));
        Alloc* ap = d.allocs.back().get();
        d.by_host[(uint64_t) (uintptr_t) p] = ap;
        d.by_dev[ap->addr] = ap;
        d.host_imports[(uint64_t) (uintptr_t) p] = ap->addr;
        ++done;
    }
    if (done == 0) return fail(kErrorUnsupported);
    return kSuccess;
}

Error host_unregister(void* p) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Ctx& c = ctx();
    for (auto& dp : c.devs) {
        Device& d = *dp;
        auto it = d.by_host.find((uint64_t) (uintptr_t) p);
        if (it == d.by_host.end() || it->second == nullptr || !it->second->imported) continue;
        Alloc* a = it->second;
        for (Span& s : a->spans) {                       // the pages belong to the caller: only the views go away
            if (s.block->buf != VK_NULL_HANDLE) vkDestroyBuffer(d.dev, s.block->buf, nullptr);
            if (s.block->mem != VK_NULL_HANDLE) vkFreeMemory(d.dev, s.block->mem, nullptr);
            s.block->buf = VK_NULL_HANDLE; s.block->mem = VK_NULL_HANDLE; s.block->map = nullptr; s.block->live = 0;
        }
        d.host_imports.erase((uint64_t) (uintptr_t) p);
        destroy_alloc(d, a);
    }
    // Only the views go away, exactly as CUDA defines cudaHostUnregister, and the range stays the caller's.  One
    // caveat this port cannot fix for a caller: the pages an import had are kept in this runtime's own free list when
    // THEY were ours (see host_pages_release), but cudaHostRegister can point at memory the caller allocated, and on
    // this driver a range that was imported is safest left alone afterwards - expert_source.cpp keeps every registered
    // window registered for the window's lifetime, which is the pattern this hardware wants.
    (void) p;
    return kSuccess;
}

Error memset_device(void* p, int value, size_t bytes) {
    Device* d = current_device("cudaMemset");
    if (d == nullptr) return fail(kErrorUnsupported);
    return fill_region(*d, stream_of(*d, nullptr), p, value, bytes, true);
}

Error memset_async(void* p, int value, size_t bytes, Stream s) {
    Device* d = current_device("cudaMemsetAsync");
    if (d == nullptr) return fail(kErrorUnsupported);
    return fill_region(*d, stream_of(*d, s), p, value, bytes, false);
}

Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind k) {
    Device* d = current_device("cudaMemcpy");
    if (d == nullptr) return fail(kErrorUnsupported);
    return copy_region(*d, stream_of(*d, nullptr), dst, src, bytes, k, true);
}

Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind k, Stream s) {
    Device* d = current_device("cudaMemcpyAsync");
    if (d == nullptr) return fail(kErrorUnsupported);
    return copy_region(*d, stream_of(*d, s), dst, src, bytes, k, false);
}

Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                      CopyKind k, Stream s) {
    Device* d = current_device("cudaMemcpy2DAsync");
    if (d == nullptr) return fail(kErrorUnsupported);
    StreamImpl* st = stream_of(*d, s);
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (width == 0 || height == 0) return kSuccess;
    Side a, b;
    if (!side_of(*d, dst, height * dpitch, a) || !side_of(*d, src, height * spitch, b)) return fail(kErrorInvalidValue);
    if (a.staged) std::memcpy(a.cpu, src, height * spitch);
    const uint64_t w = round_up(width, 4);                       // whole 4-byte regions, same rule as copy_region
    if (a.off + (height - 1) * dpitch + w > a.block->size || b.off + (height - 1) * spitch + w > b.block->size) {
        note("2d-fit", "a %zux%zu copy with pitches %llu / %llu and row width %llu (rounded to %llu) does not fit in "
                       "the buffers it was given\n", height, width, (unsigned long long) dpitch,
             (unsigned long long) spitch, (unsigned long long) width, (unsigned long long) w);
        return fail(kErrorInvalidValue);
    }
    for (size_t y = 0; y < height; ++y) {                       // one region per row: Vulkan has no 2D copy
        VkBufferCopy cp{b.off + y * spitch, a.off + y * dpitch, w};   // src row, dst row (b is the source side)
        vkCmdCopyBuffer(record_cb(st), b.buf, a.buf, 1, &cp);
    }
    st->copies++;
    nudge_submit(st);
    if (a.staged) {
        d->staging_pending = true; d->staging_stream = st; d->staging_value = st->signaled;
        wait_timeline(*d, st->tl, st->signaled, "cudaMemcpy2DAsync");
        std::memcpy(dst, a.cpu, height * dpitch);
    }
    return kSuccess;
}

// CUDA's peer copy between two cards.  There is no NVLink here and Vulkan has no peer API, so the hand-off is staged
// through host memory - the same thing the SYCL split does (src/core/expert_source.cpp:3424 "the copy is always
// host-staged").  The engine's own multi-GPU policy is unchanged; only the path is slower, and it says so once.
Error memcpy_peer_async(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, Stream s) {
    Ctx& c = ctx();
    if (!c.ok || src_dev < 0 || dst_dev < 0 || (size_t) src_dev >= c.devs.size() || (size_t) dst_dev >= c.devs.size())
        return fail(kErrorInvalidValue);
    if (src_dev == dst_dev) return memcpy_async(dst, src, bytes, kDeviceToDevice, s);
    note("peer-copy", "cudaMemcpyPeerAsync(%d -> %d, %llu MiB): Vulkan has no peer access, so this port stages the "
                      "bytes through host memory (as the SYCL split does).  The expert_cache only asks for a peer "
                      "device with NVLink, so this is not the path the cache plans for.\n", src_dev, dst_dev,
         (unsigned long long) (bytes / kMiB));
    std::vector<uint8_t> tmp(bytes);
    Device& sd = *c.devs[(size_t) src_dev];
    Device& dd = *c.devs[(size_t) dst_dev];
    StreamImpl* ss = stream_of(sd, nullptr);
    if (copy_region(sd, ss, tmp.data(), src, bytes, kDeviceToHost, true) != kSuccess) return fail(kErrorInvalidValue);
    StreamImpl* ds = stream_of(dd, s);
    return copy_region(dd, ds, dst, tmp.data(), bytes, kHostToDevice, false);
}

Error memcpy_batch_async(void* const* dst, const void* const* src, const size_t* bytes, size_t n, void* attr,
                         size_t* first_error, int num_attrs, Stream s) {
    (void) attr; (void) num_attrs;
    Device* d = current_device("cudaMemcpyBatchAsync");
    if (d == nullptr) return fail(kErrorUnsupported);
    if (dst == nullptr || src == nullptr || bytes == nullptr) return fail(kErrorInvalidValue);
    StreamImpl* st = stream_of(*d, s);
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    bool opened = false;
    for (size_t i = 0; i < n; ++i) {
        if (bytes[i] == 0) continue;
        const VkCommandBuffer cb = record_cb(st);         // every copy in the SAME chain slot ...
        if (cb == VK_NULL_HANDLE) return fail(kErrorInvalidValue);
        opened = true;
        if (copy_region(*d, st, dst[i], src[i], bytes[i], kCopyDefault, false, cb) != kSuccess) {
            if (first_error) *first_error = i;           // CUDA reports the index of the first failing copy
            nudge_submit(st);
            return fail(kErrorInvalidValue);
        }
    }
    if (opened) nudge_submit(st);                         // ... and one submit carries all of them
    return kSuccess;
}

// ---- streams -------------------------------------------------------------------------------

Error stream_create(Stream* s, unsigned flags) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaStreamCreate");
    if (d == nullptr) return fail(kErrorUnsupported);
    const uint32_t qidx = (uint32_t) d->streams.size() % (uint32_t) d->queues.size();   // round-robin over the queues
    StreamImpl* st = make_stream(*d, qidx, flags);
    if (st == nullptr) return fail(kErrorUnsupported);
    if (d->queues.size() < d->streams.size())
        note("queues", "%s offers %zu queue(s) and the engine asked for more streams; the extra streams share a queue "
                       "as a serialized submit chain (a VkQueue runs its submits in submission order), which is what "
                       "their cudaStreamWaitEvent edges order anyway.", d->props.deviceName, d->queues.size());
    if (s) *s = st;
    return kSuccess;
}

Error stream_destroy(Stream s) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (s == nullptr) return kSuccess;
    if (s->d == nullptr) return kSuccess;
    destroy_stream(*s->d, s);
    return kSuccess;
}

Error stream_synchronize(Stream s) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaStreamSynchronize");
    if (d == nullptr) return fail(kErrorUnsupported);
    StreamImpl* st = stream_of(*d, s);
    drain_readback(*d);
    nudge_submit(st);
    return wait_timeline(*d, st->tl, st->signaled, "cudaStreamSynchronize");
}

Error stream_query(Stream s) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaStreamQuery");
    if (d == nullptr) return fail(kErrorUnsupported);
    StreamImpl* st = stream_of(*d, s);
    nudge_submit(st);
    uint64_t now = 0;
    if (vkGetSemaphoreCounterValue(d->dev, st->tl, &now) != VK_SUCCESS) return fail(kErrorUnknown);
    return now >= st->signaled ? kSuccess : kNotReady;
}

Error stream_priority_range(int* lowest, int* highest) {
    if (lowest) *lowest = 0;      // CUDA's buckets, mapped onto the queue priorities open_device asked for
    if (highest) *highest = -2;   // -1 and -2 get the queues created with the higher priority
    return kSuccess;
}

Error stream_is_capturing(Stream s, int* status) {
    Device* d = current_device("cudaStreamIsCapturing");
    if (d == nullptr) return fail(kErrorUnsupported);
    StreamImpl* st = stream_of(*d, s);
    if (status) *status = st->rec != nullptr ? 1 : 0;    // cudaStreamCaptureStatusActive / None
    return kSuccess;
}

Error launch_host_func(Stream s, HostFn fn, void* arg) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaLaunchHostFunc");
    if (d == nullptr) return fail(kErrorUnsupported);
    if (fn == nullptr) return fail(kErrorInvalidValue);
    StreamImpl* st = stream_of(*d, s);
    if (st->rec != nullptr) {                              // inside a capture: a node, run at each launch
        NodeImpl* n = new NodeImpl;
        n->type = (int) cudaGraphNodeTypeHost;
        n->fn = fn; n->fn_arg = arg;
        st->rec->nodes.push_back(n);
        note("host-fn", "cudaLaunchHostFunc was recorded inside a captured graph.  Vulkan has no host node inside a "
                        "command buffer, so this port runs it at each cudaGraphLaunch, before the buffer is "
                        "submitted, not interleaved with the device work in it.");
        return kSuccess;
    }
    nudge_submit(st);                                      // after everything the stream already recorded
    if (wait_timeline(*d, st->tl, st->signaled, "cudaLaunchHostFunc") != kSuccess) return fail(kErrorLaunchFailure);
    fn(arg);
    return kSuccess;
}

// ---- events: timeline semaphores ------------------------------------------------------------

Error event_create(Event* e, unsigned flags) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaEventCreate");
    if (d == nullptr) return fail(kErrorUnsupported);
    if (flags & kEventBlockingSync)
        note("ev-sync", "cudaEventBlockingSync: vkWaitSemaphores already blocks the host thread in %llu s slices, so "
                        "there is nothing to change.", (unsigned long long) (kWaitSliceNs / 1000000000ull));
    EventImpl* ev = new EventImpl;
    ev->dev = d;
    VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;          // an event IS a timeline value (core 1.2)
    ti.initialValue = 0;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    si.pNext = &ti;
    const VkResult er = vkCreateSemaphore(d->dev, &si, nullptr, &ev->sem);
    if (er != VK_SUCCESS) {
        delete ev;
        note("ev", "vkCreateSemaphore(VK_SEMAPHORE_TYPE_TIMELINE) for an event on %s said %s\n",
             d->props.deviceName, vk_name(er));
        return fail(kErrorUnsupported);
    }
    if (e) *e = ev;
    return kSuccess;
}

Error event_destroy(Event e) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (e == nullptr) return kSuccess;
    if (e->dev != nullptr && e->sem != VK_NULL_HANDLE) vkDestroySemaphore(e->dev->dev, e->sem, nullptr);
    delete e;
    return kSuccess;
}

Error event_record(Event e, Stream s) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaEventRecord");
    if (d == nullptr || e == nullptr) return fail(kErrorInvalidValue);
    StreamImpl* st = stream_of(*d, s);
    const uint64_t v = e->value + 1;                       // monotone per event, as CUDA's re-recording is
    if (st->rec != nullptr) {                              // inside a capture: resolved to a value at each launch
        NodeImpl* n = new NodeImpl;
        n->type = (int) cudaGraphNodeTypeEventRecord;
        n->ev = e;
        st->rec->nodes.push_back(n);
        e->value = v;
        return kSuccess;
    }
    if (st->open == nullptr) {
        // Nothing recorded on this stream: the event is complete now, and vkSignalSemaphore is exactly that - a host
        // side signal of a timeline value.
        VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        si.semaphore = e->sem;
        si.value = v;
        const VkResult r = vkSignalSemaphore(d->dev, &si);          // THE RECORD SIDE, host-signalled
        if (r != VK_SUCCESS) { note("ev-record", "vkSignalSemaphore said %s", vk_name(r)); return fail(kErrorLaunchFailure); }
        e->value = v;
        return kSuccess;
    }
    st->signals.push_back(std::make_pair(e->sem, v));      // the signal entry the next submit carries
    e->value = v;
    nudge_submit(st);
    return kSuccess;
}

Error event_query(Event e) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (e == nullptr) return fail(kErrorInvalidValue);
    if (e->value == 0) return kSuccess;                    // a fresh CUDA event is "complete"
    uint64_t now = 0;
    if (vkGetSemaphoreCounterValue(e->dev->dev, e->sem, &now) != VK_SUCCESS) return fail(kErrorUnknown);
    return now >= e->value ? kSuccess : kNotReady;
}

Error event_synchronize(Event e) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (e == nullptr) return fail(kErrorInvalidValue);
    if (e->value == 0) return kSuccess;
    return wait_timeline(*e->dev, e->sem, e->value, "cudaEventSynchronize");
}

Error event_elapsed_ms(float* ms, Event a, Event b) {
    note("elapsed", "cudaEventElapsedTime has no Vulkan equivalent for a timeline semaphore: a timeline value carries "
                    "no timestamp.  What this port will use instead is vkCmdWriteTimestamp into a host-visible buffer "
                    "read back after the submit (the probe measures dispatch costs that way); until then the engine's "
                    "own wall clock is the honest answer.");
    if (ms) *ms = 0.0f;
    (void) a; (void) b;
    return fail(kErrorUnsupported);
}

Error stream_wait_event(Stream s, Event e, unsigned flags) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaStreamWaitEvent");
    if (d == nullptr) return fail(kErrorUnsupported);
    if (e == nullptr) return fail(kErrorInvalidValue);
    if (flags != 0)
        note("ev-wait", "cudaStreamWaitEvent flags (cudaEventWaitExternal) have no Vulkan equivalent; the wait is the "
                        "ordinary timeline dependency.");
    StreamImpl* st = stream_of(*d, s);
    if (st->rec != nullptr) {                              // inside a capture: a node, value resolved at launch
        NodeImpl* n = new NodeImpl;
        n->type = (int) cudaGraphNodeTypeWaitEvent;
        n->ev = e;
        st->rec->nodes.push_back(n);
        return kSuccess;
    }
    if (e->value == 0) return kSuccess;                    // nothing recorded yet: CUDA has nothing to wait for either
    st->waits.push_back(std::make_pair(e->sem, e->value)); // the WAIT entry the next submit carries
    return kSuccess;
}

// ---- graphs: one command buffer, recorded once, re-submitted per window ---------------------

Error begin_capture(Stream s) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaStreamBeginCapture");
    if (d == nullptr) return fail(kErrorUnsupported);
    StreamImpl* st = stream_of(*d, s);
    if (st->rec != nullptr) return fail(kErrorInvalidValue);
    nudge_submit(st);                                      // a capture starts from a drained stream
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = st->pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d->dev, &ai, &cb) != VK_SUCCESS) return fail(kErrorUnsupported);
    GraphImpl* g = new GraphImpl;
    g->dev = d;
    g->cb = cb;
    g->pool = st->pool;                                     // the pool its buffer is freed back into
    bool oom = false;
    g->param_alloc = sub_alloc(*d, 64 * kKiB, kHostWant, d->align_uniform, &oom);   // captured dispatches' parameters
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) { delete g; return fail(kErrorInvalidValue); }
    st->rec = g;
    return kSuccess;
}

Error end_capture(Stream s, Graph* g) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* d = current_device("cudaStreamEndCapture");
    if (d == nullptr) return fail(kErrorUnsupported);
    StreamImpl* st = stream_of(*d, s);
    if (st->rec == nullptr) return fail(kErrorInvalidValue);
    if (vkEndCommandBuffer(st->rec->cb) != VK_SUCCESS) { note("capture", "vkEndCommandBuffer failed on the captured graph"); return fail(kErrorInvalidValue); }
    st->rec->ended = true;
    if (g) *g = st->rec;
    st->rec = nullptr;
    return kSuccess;
}

Error graph_nodes(Graph g, Node* nodes, size_t* n) {
    if (g == nullptr || n == nullptr) return fail(kErrorInvalidValue);
    if (nodes == nullptr) { *n = g->nodes.size(); return kSuccess; }   // cudaGraphGetNodes' count-only form
    size_t i = 0;
    for (; i < *n && i < g->nodes.size(); ++i) nodes[i] = g->nodes[i];
    *n = i;
    return kSuccess;
}

Error graph_node_type(Node n, int* type) {
    if (n == nullptr || type == nullptr) return fail(kErrorInvalidValue);
    *type = n->type;
    return kSuccess;
}

Error graph_node_kernel_name(Node n, const char** name) {
    if (n == nullptr || name == nullptr) return fail(kErrorInvalidValue);
    *name = n->name.c_str();
    return kSuccess;
}

Error graph_node_params(Node n, void** func, Dim3* g, Dim3* b, size_t* shared_mem) {
    if (n == nullptr) return fail(kErrorInvalidValue);
    if (func) *func = nullptr;                             // no host function pointer exists for a SPIR-V entry point
    if (g) *g = n->grid;
    if (b) *b = n->block;
    if (shared_mem) *shared_mem = n->shared;
    return kSuccess;
}

Error graph_edges(Graph g, Node* from, Node* to, int* edge_type, size_t* n) {
    if (g == nullptr || n == nullptr) return fail(kErrorInvalidValue);
    const size_t e = g->nodes.size() > 1 ? g->nodes.size() - 1 : 0;    // the recorded order IS the dependency
    if (from == nullptr || to == nullptr) { *n = e; return kSuccess; }
    size_t i = 0;
    for (; i < *n && i < e; ++i) {
        from[i] = g->nodes[i];
        to[i] = g->nodes[i + 1];
        if (edge_type) edge_type[i] = (int) cudaGraphDependencyTypeDefault;   // Vulkan has no programmatic (PDL) edge
    }
    *n = i;
    return kSuccess;
}

Error graph_instantiate(GraphExec* x, Graph g) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (g == nullptr || g->cb == VK_NULL_HANDLE) return fail(kErrorInvalidValue);
    GraphExecImpl* e = new GraphExecImpl;
    e->dev = g->dev;
    e->graph = g;
    e->cb = g->cb;                                       // the same VkCommandBuffer the capture recorded
    e->reencode = env_on("STRATA_VULKAN_GRAPH_REUPLOAD", false);
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g->dev->dev, &fi, nullptr, &e->fence) != VK_SUCCESS) { delete e; return fail(kErrorUnsupported); }
    // The per-launch parameter bytes of the captured dispatches go in one host-visible host-coherent block, written
    // afresh by every replay (record_dispatch's window slot).  The engine's per-window values live in device buffers
    // it rewrites before each launch, so a replay needs no graph update - which is why Vulkan's missing
    // cudaGraphExecUpdate is not a hole here: src/core/verify.cpp:1947-2006 never calls it.
    uint64_t params = 0;
    for (const NodeImpl* nn : g->nodes) if (nn->type == (int) cudaGraphNodeTypeKernel) params += nn->params.size() + 256;
    if (params != 0) {
        bool oom = false;
        e->window_alloc = sub_alloc(*g->dev, params, kHostWant, g->dev->align_uniform, &oom);
        if (e->window_alloc != nullptr) e->window_bytes = e->window_alloc->bytes;
    }
    g->refs++;
    if (x) *x = e;
    return kSuccess;
}

Error graph_launch(GraphExec x, Stream s) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (x == nullptr || x->graph == nullptr || x->cb == VK_NULL_HANDLE) return fail(kErrorInvalidValue);
    Device& d = *x->dev;
    drain_readback(d);                                     // a staged read-back from an earlier launch is handed over now
    StreamImpl* st = stream_of(d, s);
    if (x->inflight) {                                     // the buffer is in use until its fence retires
        vkWaitForFences(d.dev, 1, &x->fence, VK_TRUE, kFenceWaitNs);
        vkResetFences(d.dev, 1, &x->fence);               // ... and the fence goes back to unsignaled before the next
        x->inflight = false;                              // submit of it (VUID-vkQueueSubmit-fence-00063)
    }
    nudge_submit(st);                                      // whatever the stream already recorded goes first
    // A captured event record / wait becomes a timeline value known at launch time, which is CUDA's semantics too:
    // the node waits for "the event as recorded", not for a frozen number.
    for (const NodeImpl* n : x->graph->nodes) {
        if (n->type == (int) cudaGraphNodeTypeWaitEvent && n->ev != nullptr && n->ev->value != 0)
            st->waits.push_back(std::make_pair(n->ev->sem, n->ev->value));
        if (n->type == (int) cudaGraphNodeTypeEventRecord && n->ev != nullptr) {
            n->ev->value = n->ev->value + 1;
            st->signals.push_back(std::make_pair(n->ev->sem, n->ev->value));
        }
        if (n->type == (int) cudaGraphNodeTypeHost && n->fn != nullptr) n->fn(n->fn_arg);
    }
    // Replay: vkResetCommandBuffer, re-record the node list into the SAME VkCommandBuffer, submit it again.  The first
    // launch submits what the capture recorded, exactly as recorded.
    if (x->launches != 0 || x->reencode) {
        if (reencode_graph(d, x) != kSuccess) return fail(kErrorInvalidValue);
    }
    if (submit_work(st, x->cb, x->fence) != VK_SUCCESS) return fail(kErrorLaunchFailure);
    x->inflight = true;
    x->launches++;
    return kSuccess;
}

Error graph_upload(GraphExec x, Stream s) {
    (void) s;
    if (x == nullptr) return fail(kErrorInvalidValue);
    note("graph-upload", "cudaGraphUpload has no Vulkan equivalent: the cost of a replay is the submit, and the "
                         "replay re-encodes the captured node list into the one VkCommandBuffer the capture recorded "
                         "(STRATA_VULKAN_GRAPH_REUPLOAD=1 makes even the first launch re-encode, which is how the "
                         "re-encode path is exercised).");
    return kSuccess;
}

void free_graph(Device& d, GraphImpl* g) {
    for (NodeImpl* n : g->nodes) delete n;
    g->nodes.clear();
    if (g->param_alloc != nullptr) destroy_alloc(d, g->param_alloc);
    if (g->cb != VK_NULL_HANDLE && g->pool != VK_NULL_HANDLE) vkFreeCommandBuffers(d.dev, g->pool, 1, &g->cb);
    delete g;
}

Error graph_destroy(Graph g) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (g == nullptr) return kSuccess;
    if (--g->refs > 0) return kSuccess;                    // an exec still replays this buffer
    free_graph(*g->dev, g);
    return kSuccess;
}

Error graph_exec_destroy(GraphExec x) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    if (x == nullptr) return kSuccess;
    Device& d = *x->dev;
    drain_readback(d);
    if (x->fence != VK_NULL_HANDLE) {
        if (x->inflight) vkWaitForFences(d.dev, 1, &x->fence, VK_TRUE, kFenceWaitNs);   // its last submit is over
        vkDestroyFence(d.dev, x->fence, nullptr);
    }
    if (x->window_alloc != nullptr) destroy_alloc(d, x->window_alloc);
    x->sets.clear();                                       // the sets themselves live in each pipeline's own pool
    GraphImpl* g = x->graph;
    if (g != nullptr && --g->refs == 0) free_graph(d, g);
    delete x;
    return kSuccess;
}

// ---- the driver API's virtual memory, and the launch face ------------------------------------

Error driver_entry_point(const char* name, DriverFn* out) {
    if (out) *out = nullptr;
    // CUDA's cuMem* VMM is what src/core/vmm.cpp asks for by name.  Vulkan's answer is a different shape (one
    // buffer per address range, re-bound with vkBindBufferMemory2), and it is not built yet, so the honest answer is
    // "not found" - which vmm.cpp:33-36 already treats as "the VMM path is off", leaving --vram-elastic off rather
    // than pretending to be CUDA.  Deliberately not recorded as the last error: it is a question, not a failure.
    note("vmm", "cudaGetDriverEntryPoint(\"%s\"): this port has no cuMem* answer yet.  The Vulkan equivalent is one "
                "VkBuffer per address range re-bound with vkBindBufferMemory2; until Phase 1's segmented arena is "
                "built, src/core/vmm.cpp sees \"not found\" and runs without --vram-elastic.", name ? name : "");
    return kErrorUnsupported;
}

Error launch_kernel(const char* name, LaunchConfig cfg, const void* args, size_t args_bytes) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Device* dp = current_device("launch_kernel");
    if (dp == nullptr) return fail(kErrorUnsupported);
    Device& d = *dp;
    Pipeline* p = find_pipeline(d, name);
    if (p == nullptr) {
        note("launch", "launch_kernel(\"%s\"): no SPIR-V pipeline is registered under that name on %s.  A Phase 2 "
                       "kernel registers with strata::gpu::register_pipeline (its .comp under vulkan/shaders/, built "
                       "by vulkan/CMakeLists.txt); until then nothing is dispatched and nothing is pretended.",
             name ? name : "", d.props.deviceName);
        return fail(kErrorUnsupported);
    }
    StreamImpl* st = stream_of(d, cfg.stream);
    if (st->rec != nullptr) {                       // recording a capture: the node list AND the vkCmd* in the buffer
        NodeImpl* n = new NodeImpl;
        n->type = (int) cudaGraphNodeTypeKernel;
        n->name = name ? name : "";
        n->grid = cfg.grid; n->block = cfg.block; n->shared = cfg.shared_mem;
        if (args && args_bytes) n->params.assign((const uint8_t*) args, (const uint8_t*) args + args_bytes);
        st->rec->nodes.push_back(n);
    }
    return record_dispatch(d, st, *p, cfg, args, args_bytes);
}

Error kernel_name(const void* entry_point, const char** name) {
    if (name == nullptr) return fail(kErrorInvalidValue);
    Ctx& c = ctx();
    for (auto& dp : c.devs)
        for (auto& kv : dp->pipes)
            if ((const void*) &kv.second == (const void*) entry_point) { *name = kv.second.name.c_str(); return kSuccess; }
    note("func-name", "cudaFuncGetName(%p): this port has no host function pointer for a kernel - an entry point is a "
                      "SPIR-V name in the pipeline cache, so only a registered pipeline answers.\n", entry_point);
    return fail(kErrorUnsupported);
}

// Where a Phase 2 kernel plugs in: build the pipeline for one SPIR-V entry point on every device this process opened.
// The parameter block a CUDA kernel's arguments become, plus how many uint64 device addresses in it become storage
// bindings (one binding per pointer argument, with the span's own offset - possibly past 4 GiB - and a range clamped
// by bind_span).
Error register_pipeline(const PipelineSpec& spec) {
    std::lock_guard<std::recursive_mutex> lk(g_gate);
    Ctx& c = ctx();
    if (!c.ok) return fail(kErrorUnsupported);
    Error last = kSuccess;
    for (auto& dp : c.devs) {
        std::string why;
        last = build_pipeline(*dp, spec, why);
        if (last != kSuccess) note("register-pipe", "%s: %s (%s)\n", dp->props.deviceName, why.c_str(), spec.name);
    }
    return last;
}

}  // namespace strata::gpu
