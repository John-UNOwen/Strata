// vulkan/include/strata/gpu/runtime.hpp - the device runtime face of the Vulkan build (Phase 1).
//
// The shared host files keep calling the CUDA runtime names; on Vulkan those names come from
// vulkan/include/vulkan_compat/cuda_runtime.h (first on the include path, as include/strata/hip_compat is on HIP and
// metal_compat is on Metal), and every one of them forwards to a function declared here.  The shape of this namespace
// is metal/include/strata/gpu/runtime.hpp's, plus what the ../src host tree needs that Metal's runtime does not have
// (the driver API's virtual-memory functions - src/core/vmm.cpp and src/core/expert_cache.cpp - and the launch face).
//
// The signatures and the meaning of each call are those of the CUDA runtime call of the same purpose.  Each group
// below says which Vulkan mechanism implements it; vulkan/src/gpu/runtime.cpp is where each one is DEFINED.
//
// What each group is implemented with, and what is deliberately NOT answered, is written into vulkan/src/gpu/runtime.cpp
// at the function that answers it.  Two families are answered with a named refusal rather than a number, because
// Vulkan has no field for them and inventing one is how a hole gets hidden: the clock / bus-width / compute-unit-count
// attributes (device_sm_count, device_clock_khz, the cudaDevAttr cases 13/16/36/37) and the cuMem* driver API
// (driver_entry_point).  src/program/generate.cpp:3828-3832 already survives multiProcessorCount == 0, and
// src/core/vmm.cpp:33-36 already treats "entry point not found" as "the VMM path is off".
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::gpu {

// ---- vocabulary -------------------------------------------------------------------------
// CUDA's cudaError_t values, kept as CUDA spells them: the host code compares against cudaSuccess and prints
// error_string(), and nothing in ../src branches on a numeric code.
enum Error : int {
    kSuccess = 0,
    kErrorInvalidValue = 1,
    kErrorMemoryAllocation = 2,
    kErrorLaunchFailure = 719,          // a vkQueueSubmit that came back failed, or a device-lost event
    kErrorPeerAccessAlreadyEnabled = 704,
    kNotReady = 600,
    kErrorUnsupported = 801,
    kErrorUnknown = 999,
};

struct StreamImpl;
struct EventImpl;
struct GraphImpl;
struct GraphExecImpl;
struct NodeImpl;      // one recorded node: its kind, its entry point's name, its parameters
struct AllocationImpl;

using Stream = StreamImpl*;      // nullptr = the default (legacy) stream, as on CUDA: one command queue per stream
using Event = EventImpl*;        // a (VkDevice, timeline semaphore, value) pair: timeline semaphores are core 1.2
using Graph = GraphImpl*;        // one recorded VkCommandBuffer buffer, captured from the stream's submissions
using GraphExec = GraphExecImpl*;  // the same buffer ready to resubmit; CUDA's "update the graph" is not used here
using Node = NodeImpl*;          // cudaGraphNode_t
using HostFn = void (*)(void*);

// The grid / block a launch carries: x,y,z workgroups (CUDA's dim3).  Declared with the vocabulary because the graph
// node listing below reports a recorded launch's grid and block.
struct Dim3 { unsigned x = 1, y = 1, z = 1; };

// The direction of a copy, in Vulkan terms:
//   kHostToDevice   host-visible (or VK_EXT_external_memory_host imported) buffer -> device-local buffer, vkCmdCopyBuffer
//   kDeviceToHost   device-local buffer -> host-visible buffer, vkCmdCopyBuffer, then the host waits on the timeline
//   kDeviceToDevice one vkCmdCopyBuffer inside the stream's command buffer (no host round trip, no P2P API: Vulkan
//                   copies inside one device; across two cards the engine hands off on the host, as the SYCL split does)
//   kCopyDefault    direction taken from the two pointers' memory types (VkMemoryPropertyFlags)
enum CopyKind : int { kHostToHost = 0, kHostToDevice = 1, kDeviceToHost = 2, kDeviceToDevice = 3, kCopyDefault = 4 };

// The fields of cudaDeviceProp that ../src reads.  Filled from VkPhysicalDeviceProperties / VkPhysicalDeviceMemory
// Properties: name from deviceName, totalGlobalMem from the device-local heap's size, warpSize from the subgroup
// properties' subgroupSize, major/minor from apiVersion (see the compat header's note), maxSharedMemoryPerOptin from
// maxComputeLocalSize.  multiProcessorCount, memoryClockRate, memoryBusWidth and l2CacheSize stay 0: no
// VkPhysicalDeviceLimits field reports a compute-unit count, a clock rate, a bus width or an L2 size, and the Vulkan
// vendor extensions that come close (VK_AMDX_shader_enqueue's activeComputeUnitCount, VK_EXT...simdPerComputeUnit) are
// not on this driver.  A 0 from here means "unknown", and src/program/generate.cpp:3828-3832 already handles it.
struct DeviceProp {
    char name[256] = {};
    char gcnArchName[256] = {};       // unused on Vulkan (HIP's arch string); kept so the host code compiles unchanged
    int major = 0, minor = 0, warpSize = 32, multiProcessorCount = 0;
    size_t totalGlobalMem = 0;
    int memoryClockRate = 0, memoryBusWidth = 0, l2CacheSize = 0;
    int maxSharedMemoryPerOptin = 0;  // cudaDevAttrMaxSharedMemoryPerBlockOptin -> maxComputeLocalSize (32768 B per
                                      // workgroup on both Arcs); the pipeline-wide field is maxComputeSharedMemorySize
    // CUDA 12.4's cudaDeviceProp::luid, which src/core/pinned.cu:368 hands straight to
    // strata::platform::gpu_shared_memory_budget (src/platform/memory.cpp:112-131: it enumerates DXGI adapters and
    // memcmps these 8 bytes against each AdapterLuid - the layout of Windows' LUID, DWORD Part1 then LONG Part2, and
    // what include/strata/platform/memory.hpp:35 promises: "8 bytes at `luid` (cudaDeviceProp::luid)").  CUDA spells
    // the member so `p.luid` IS those bytes, so it is an array here as well.  Vulkan answers the same question
    // directly: VkPhysicalDeviceIDProperties (vkGetPhysicalDeviceProperties2) carries deviceLUID[VK_LUID_SIZE], which is
    // 8 bytes, plus deviceNodeMask and deviceLUIDValid - documented as the physical device's LUID, the same identifier
    // DXGI matches on.  get_device_properties copies all 8 bytes when deviceLUIDValid is set (VK_LUID_SIZE == 8 makes
    // the array the same width as CUDA's, which is why this struct needs no padding).
    unsigned char luid[8] = {};             // CUDA spells it `char luid[8]` ("8-byte locally unique identifier.
                                            // Value is undefined on TCC and non-Windows platforms")
    unsigned int luidDeviceNodeMask = 0;    // CUDA's `unsigned int luidDeviceNodeMask`
};

inline constexpr unsigned kHostAllocDefault = 0;
inline constexpr unsigned kHostAllocPortable = 1;
inline constexpr unsigned kHostAllocMapped = 2;
inline constexpr unsigned kHostAllocWriteCombined = 4;
inline constexpr unsigned kHostRegisterDefault = 0;
inline constexpr unsigned kHostRegisterPortable = 1;
inline constexpr unsigned kHostRegisterMapped = 2;
inline constexpr unsigned kHostRegisterReadOnly = 8;
inline constexpr unsigned kEventDefault = 0;
inline constexpr unsigned kEventBlockingSync = 1;
inline constexpr unsigned kEventDisableTiming = 2;
inline constexpr unsigned kStreamNonBlocking = 1;
inline constexpr unsigned kStreamLegacy = 0;

// Phase 0 measured no host-polled doorbell on this driver (P5 failed in all four memory flavors), so nothing in this
// port may assume the GPU sees a flag the host raised while its own dispatch runs: waits are submits.
struct Caps {
    bool unified_memory = false;        // Vulkan here is not unified: device-local buffers, explicit copies
    bool host_mapped_waits = false;     // the doorbell replacement is a timeline semaphore or a nudge submit
};
Caps caps();
void* host_view(const void* device_pointer);   // the CPU address of a host-visible allocation (nullptr for device-local)

// ---- errors, devices ---------------------------------------------------------------------
Error get_last_error();
Error peek_last_error();
const char* error_string(Error e);
const char* error_name(Error e);
Error get_device(int* dev);
Error set_device(int dev);
Error device_count(int* n);
Error device_synchronize();
Error mem_get_info(size_t* free_b, size_t* total_b);   // VK_EXT_memory_budget per card, minus this process's own usage
Error get_device_properties(DeviceProp* p, int dev);
Error device_integrated(int* v, int dev);              // VkPhysicalDeviceProperties::deviceType == INTEGRATED_DEVICE
Error device_sm_count(int* v, int dev);                // no Vulkan answer: VkPhysicalDeviceLimits has no unit count
Error device_clock_khz(int* v, int dev);
Error device_attr_int(int* v, int attr, int dev);      // the cudaDeviceGetAttribute cases the host code asks for
Error runtime_version(int* v);
Error device_can_access_peer(int* v, int dev, int peer);   // Vulkan: two VkDevices, no peer API -> false
Error device_enable_peer_access(int peer, unsigned flags);
Error device_set_flags(unsigned flags);                   // spin / yield / blocking-sync: no Vulkan equivalent; recorded
Error device_map_host();                                  // host-coherent mapped memory is always available here

// ---- memory and copies -------------------------------------------------------------------
Error malloc_device_raw(void** p, size_t bytes);      // a buffer over the sub-allocator's VkDeviceMemory blocks,
                                                      // with a buffer-device-address on every allocation
template <class T> inline Error malloc_device(T** p, size_t bytes) { return malloc_device_raw((void**) (void*) p, bytes); }
Error free_device(void* p);
Error host_alloc_raw(void** p, size_t bytes, unsigned flags);   // host-visible + host-coherent; Mapped means the
                                                                // same pointer is the device address (P3: the import
                                                                // into both cards' devices works)
template <class T> inline Error host_alloc(T** p, size_t bytes, unsigned flags) {
    return host_alloc_raw((void**) (void*) p, bytes, flags);
}
Error free_host(void* p);
template <class T> inline Error malloc_host(T** p, size_t bytes) { return host_alloc_raw((void**) (void*) p, bytes, 0); }
Error host_device_pointer_raw(void** dev, void* host);
template <class T> inline Error host_device_pointer(T** dev, void* host, unsigned = 0) {
    return host_device_pointer_raw((void**) (void*) dev, host);
}
Error host_register(void* p, size_t bytes, unsigned flags);   // VK_EXT_external_memory_host import of an existing
                                                              // host allocation (Phase 0 P3: pass)
Error host_unregister(void* p);
Error memset_device(void* p, int value, size_t bytes);        // vkCmdFillBuffer
Error memset_async(void* p, int value, size_t bytes, Stream s = nullptr);   // vkCmdFillBuffer in the stream

Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind k);
Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind k, Stream s = nullptr);   // vkCmdCopyBuffer
Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                      CopyKind k, Stream s = nullptr);   // one region-to-region copy per row: no 2D copy in Vulkan
Error memcpy_peer_async(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, Stream s = nullptr);
Error memcpy_batch_async(void* const* dst, const void* const* src, const size_t* bytes, size_t n, void* attr,
                         size_t* first_error, int num_attrs, Stream s);   // n vkCmdCopyBuffers in one command buffer

// ---- streams, events, host functions -----------------------------------------------------
Error stream_create(Stream* s, unsigned flags);       // one VkQueue (or one command buffer on the device's queue)
Error stream_destroy(Stream s);
Error stream_synchronize(Stream s);                   // wait for this stream's timeline value
Error stream_query(Stream s);                         // VkTimelineSemaphoreSubmitInfo value already signalled?
Error stream_wait_event(Stream s, Event e, unsigned flags = 0);   // a wait entry (VkSubmitInfo::pWaitSemaphores with its
                                                                  // value) on the stream's next submit
Error stream_priority_range(int* lowest, int* highest);           // VkQueueSubmit priority (global priority extension)
Error stream_is_capturing(Stream s, int* status);
Error launch_host_func(Stream s, HostFn fn, void* arg);           // run it on the host at that point in the record

Error event_create(Event* e, unsigned flags);         // a timeline semaphore per event (DisableTiming skips the pair)
Error event_destroy(Event e);
Error event_record(Event e, Stream s = nullptr);      // a signal entry (VkSubmitInfo::pSignalSemaphores with its value
                                                      // in VkTimelineSemaphoreSubmitInfo) on the submit that carries the
                                                      // stream's recorded work; vkSignalSemaphore when it recorded none
Error event_query(Event e);
Error event_synchronize(Event e);                     // vkWaitSemaphores on the event's timeline value
Error event_elapsed_ms(float* ms, Event a, Event b);  // no Vulkan answer: a timeline value carries no timestamp

// ---- graphs ------------------------------------------------------------------------------
// A verify window is one VkCommandBuffer recorded once (stream capture) and resubmitted per window.  The engine keeps
// per-window values in device buffers, so a replay needs no graph update (VULKAN-BACKEND-PLAN.md's mapping table).
Error begin_capture(Stream s);
Error end_capture(Stream s, Graph* g);
Error graph_nodes(Graph g, Node* nodes, size_t* n);   // nodes == nullptr: count only (cudaGraphGetNodes' two forms)
Error graph_node_type(Node n, int* type);             // cudaGraphNodeTypeKernel / Memcpy / Memset / Host / ...
Error graph_node_kernel_name(Node n, const char** name);   // verify.cpp's listing wants the entry point's name
Error graph_node_params(Node n, void** func, Dim3* grid, Dim3* block, size_t* shared_mem);   // cudaKernelNodeParams
Error graph_edges(Graph g, Node* from, Node* to, int* edge_type, size_t* n);   // edge_type: cudaGraphDependencyType;
                                                      // Vulkan has no programmatic edge (no PDL), so it is always the
                                                      // ordinary dependency, and verify.cpp's PDL listing says "none"
Error graph_instantiate(GraphExec* x, Graph g);
inline Error graph_instantiate(GraphExec* x, Graph g, unsigned long long) { return graph_instantiate(x, g); }
Error graph_launch(GraphExec x, Stream s);                    // vkQueueSubmit of the recorded buffer
Error graph_upload(GraphExec x, Stream s);
Error graph_destroy(Graph g);
Error graph_exec_destroy(GraphExec x);

// ---- the driver API's virtual memory management (src/core/vmm.cpp, src/core/expert_cache.cpp) --
// CUDA 12's cuMem* VMM is how the segmented arena (--vram-elastic) maps and unmaps tail segments.  Vulkan's answer is
// the same trick Strata already uses on the host side: reserve one address range, and bind/unbind VkDeviceMemory
// chunks into one buffer with vkBindBufferMemory2 / vkDestroyBuffer, or map a range by re-creating the buffer view.
// The names must exist with CUDA's exact signatures: vmm.cpp takes them by name through cudaGetDriverEntryPoint and
// calls them through function pointers.
using DriverFn = void*;
Error driver_entry_point(const char* name, DriverFn* out);   // name -> this port's implementation of that cuX name

// ---- the launch face ---------------------------------------------------------------------
// The grid / block / shared-memory / stream a CUDA launch carries.  Block is the workgroup size; SharedMem is
// maxComputeLocalSize in the pipeline; Stream decides which command buffer the dispatch goes into.  (Dim3 itself is
// declared with the vocabulary above.)
struct LaunchConfig {
    Dim3 grid, block;
    size_t shared_mem = 0;
    Stream stream = nullptr;
};
// One dispatch of one entry point.  CUDA passes the kernel's arguments one by one; a SPIR-V entry point takes them as
// one parameter block (a uniform buffer or push constants), so `args` is that block and `args_bytes` its size - the
// same shape the kernel headers already use for their multi-argument kernels (src/kernels/cuda/fused_gr.cu's
// gr_down_kernel(GrMulti m), kv_stream.cu's resolve_kernel(KvStreamMap m)).
Error launch_kernel(const char* name, LaunchConfig cfg, const void* args, size_t args_bytes);

// The name a node's / a function pointer's entry point has (cudaFuncGetName, verify.cpp:1963).  On Vulkan the entry
// points are SPIR-V names in the pipeline cache this runtime built, so the name comes from there, not from a host
// symbol table.
Error kernel_name(const void* entry_point, const char** name);

// Where a Phase 2 kernel plugs in.  One SPIR-V entry point (the bytes a .comp under vulkan/shaders/ becomes, through
// glslc and tools/embed_spv.py) plus what a launch of it needs: the workgroup size the SPIR-V declares, the size of
// its parameter block, and how many uint64 device addresses in that block become descriptor bindings.  register_pipeline
// builds the VkShaderModule / VkDescriptorSetLayout / VkPipelineLayout / VkPipeline on every device this process opened,
// under this name; launch_kernel(name, ...) then finds it.  A name nothing registered is refused by name.
struct PipelineSpec {
    const char* name = nullptr;             // the launch name launch_kernel is called with
    const char* entry = "main";             // the SPIR-V entry point (GLSL .comp compiles to "main")
    const uint32_t* spv = nullptr;          // the module's words
    size_t spv_words = 0;
    Dim3 workgroup{1, 1, 1};                // the local_size the SPIR-V declares
    size_t shared_mem = 0;                  // workgroup-local bytes it asks for; refused above
                                            // VkPhysicalDeviceLimits::maxComputeSharedMemorySize (49152 B on both Arcs)
    size_t param_bytes = 0;                 // its parameter block, passed as push constants: refused above this card's
                                            // maxPushConstantsSize (256 B on both Arcs) unless
                                            // STRATA_VULKAN_UNIFORM_PARAMS=1 selects the uniform-buffer path
    uint32_t storage_bindings = 0;          // uint64 device-address parameters that become storage-buffer bindings
};
Error register_pipeline(const PipelineSpec& spec);

}  // namespace strata::gpu
