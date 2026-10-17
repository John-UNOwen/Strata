// vulkan/include/vulkan_compat/cuda_runtime.h - the CUDA runtime names the host files use, on Vulkan.
//
// The HIP build's approach (include/strata/hip_compat/cuda_runtime.h maps every cudaX the host code calls to hipX)
// and the Metal build's (metal/include/metal_compat/cuda_runtime.h), applied to Vulkan: this directory goes first on
// the Vulkan build's include path, so the shared host files in ../src compile for Vulkan UNCHANGED, and the CUDA /
// HIP / SYCL builds never see this file.  Every name forwards to strata::gpu's Vulkan implementation
// (vulkan/include/strata/gpu/runtime.hpp, implemented in vulkan/src/gpu/runtime.cpp), whose semantics are CUDA's.
//
// THE INVENTORY IS THE ERROR LOG, not a wish list.  The method is metal_compat's, recorded in its comment
// (metal/include/metal_compat/cuda_runtime.h:7 - "Only the names in the Phase 1 inventory ... anything else fails to
// compile, on purpose") and in VULKAN-BACKEND-PLAN.md Phase 1 ("Get the inventory by building with an empty header and
// reading the compile errors").  Iteration 0 of this file was `#pragma once` and nothing else; the loop is recorded at
// the bottom of this file and build-vulkan\logs\iter-*.log are the transcripts.
//
// What is deliberately NOT here: anything only the CUDA *device* pass has.  <<< >>> launches, __global__ kernels, the
// shuffle / sub-group intrinsics and the fast-math builtins live in the .cu kernels this backend replaces with GLSL ->
// SPIR-V, or behind the `#if defined(__CUDACC__) || defined(__HIPCC__)` guard in the kernel headers -
// include/strata/kernels/pdl.hpp:34 keeps its launch_pdl, whose last line IS `kernel<<<grid, block, smem, stream>>>`
// (pdl.hpp:95), inside that guard for exactly that reason.  cl.exe cannot tokenize <<< at all, and nothing in the
// compiled host tree does.
#pragma once

#include <cstddef>
#include <cstdint>

#include <immintrin.h>               // MSVC's own, not a shim: ../src's CPU kernels (src/kernels/cpu/*) are x86
                                     // intrinsics on an x86 PC.  metal_compat has to shim this file because arm64 has
                                     // no immintrin.h; here the real one is used and only its GCC spelling <cpuid.h>
                                     // (src/kernels/cpu/expert.cpp:12, expert_layout.cpp:13) needs a shim - see below.
#include "strata/gpu/runtime.hpp"    // vulkan/include/strata: the Vulkan runtime this header forwards to

namespace strata::gpu::compat_detail { inline Error bad() { return kErrorUnsupported; } }

// ---- types and constants -----------------------------------------------------------------
using cudaError_t = strata::gpu::Error;
using cudaStream_t = strata::gpu::Stream;             // a stream is a VkQueue + its command buffer + a timeline value
using cudaEvent_t = strata::gpu::Event;               // an event is a timeline semaphore value (core 1.2)
using cudaGraph_t = strata::gpu::Graph;               // a graph is one recorded VkCommandBuffer
using cudaGraphExec_t = strata::gpu::GraphExec;       // ... resubmitted per launch; CUDA's graph *update* is not used
using cudaGraphNode_t = strata::gpu::Node;            // one recorded node: kind, entry point, parameters
using cudaDeviceProp = strata::gpu::DeviceProp;       // filled from VkPhysicalDeviceProperties / the memory heaps
using cudaMemcpyKind = strata::gpu::CopyKind;
using cudaHostFn_t = strata::gpu::HostFn;
using dim3 = strata::gpu::Dim3;                       // the grid / block a launch carries (x,y,z workgroups)

inline constexpr cudaError_t cudaSuccess = strata::gpu::kSuccess;
inline constexpr cudaError_t cudaErrorInvalidValue = strata::gpu::kErrorInvalidValue;
inline constexpr cudaError_t cudaErrorMemoryAllocation = strata::gpu::kErrorMemoryAllocation;
inline constexpr cudaError_t cudaErrorNotReady = strata::gpu::kNotReady;
inline constexpr cudaError_t cudaErrorLaunchFailure = strata::gpu::kErrorLaunchFailure;
inline constexpr cudaError_t cudaErrorPeerAccessAlreadyEnabled = strata::gpu::kErrorPeerAccessAlreadyEnabled;
inline constexpr cudaError_t cudaErrorStreamCaptureUnsupported = strata::gpu::kErrorUnsupported;
inline constexpr cudaError_t cudaErrorUnknown = strata::gpu::kErrorUnknown;

// The copy directions, each with the Vulkan transfer it performs.  All four are one vkCmdCopyBuffer recorded in the
// stream's command buffer; what differs is which side is host-visible.  Phase 0 P3 measured on this PC that one host
// allocation imported into BOTH cards' devices with VK_EXT_external_memory_host reads correct data in place on each,
// so HostToDevice / DeviceToHost are a copy from or into that import, not a staging dance.
inline constexpr cudaMemcpyKind cudaMemcpyHostToHost = strata::gpu::kHostToHost;          // host -> host: the CPU does it,
                                                                                          // no VkCommand at all
inline constexpr cudaMemcpyKind cudaMemcpyHostToDevice = strata::gpu::kHostToDevice;      // imported host-visible buffer
                                                                                          // -> device-local buffer
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = strata::gpu::kDeviceToHost;      // device-local buffer
                                                                                          // -> host-visible buffer, then
                                                                                          // the host waits on the timeline
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = strata::gpu::kDeviceToDevice;  // one copy, both sides
                                                                                          // device-local, same queue
inline constexpr cudaMemcpyKind cudaMemcpyDefault = strata::gpu::kCopyDefault;            // direction read from the two
                                                                                          // allocations' memory types

inline constexpr unsigned cudaHostAllocDefault = strata::gpu::kHostAllocDefault;
inline constexpr unsigned cudaHostAllocPortable = strata::gpu::kHostAllocPortable;   // readable by every VkDevice in the
                                                                                     // process: one host allocation,
                                                                                     // imported into each card (P3 pass)
inline constexpr unsigned cudaHostAllocMapped = strata::gpu::kHostAllocMapped;       // host-visible and host-coherent:
                                                                                     // the host pointer is the address a
                                                                                     // shader is given.  NOT a doorbell:
                                                                                     // Phase 0 P5 measured that a
                                                                                     // parked shader never sees a flag the
                                                                                     // host raised during its own dispatch
                                                                                     // on this driver, so waits are submits
inline constexpr unsigned cudaHostAllocWriteCombined = strata::gpu::kHostAllocWriteCombined;  // host-coherent without
                                                                                               // host-cached
inline constexpr unsigned cudaHostRegisterDefault = strata::gpu::kHostRegisterDefault;
inline constexpr unsigned cudaHostRegisterPortable = strata::gpu::kHostRegisterPortable;
inline constexpr unsigned cudaHostRegisterMapped = strata::gpu::kHostRegisterMapped;
inline constexpr unsigned cudaHostRegisterReadOnly = strata::gpu::kHostRegisterReadOnly;   // the import as read-only to
                                                                                           // the GPU
inline constexpr unsigned cudaEventDefault = strata::gpu::kEventDefault;
inline constexpr unsigned cudaEventDisableTiming = strata::gpu::kEventDisableTiming;   // a timeline value, no timestamp pair
inline constexpr unsigned cudaEventBlockingSync = strata::gpu::kEventBlockingSync;      // vkWaitSemaphores, not a host spin
inline constexpr unsigned cudaStreamNonBlocking = strata::gpu::kStreamNonBlocking;      // its own queue, no legacy join
inline constexpr unsigned cudaStreamLegacy = strata::gpu::kStreamLegacy;                // 0: the default stream

enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1,
                             cudaStreamCaptureModeRelaxed = 2 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1,
                               cudaStreamCaptureStatusInvalidated = 2 };

// CUDA's numeric attribute ids, spelled as CUDA spells them (enum cudaDeviceAttr in cuda_runtime.h: ClockRate = 13,
// MultiProcessorCount = 16, Integrated = 18, MemoryClockRate = 36, GlobalMemoryBusWidth = 37,
// MaxSharedMemoryPerBlockOptin = 97, ComputeCapabilityMajor = 75, ComputeCapabilityMinor = 76), so
// cudaDeviceGetAttribute(v, cudaDevAttrX, d) in ../src compiles unchanged; what each one answers from Vulkan is in
// strata::gpu::device_attr_int (vulkan/src/gpu/runtime.cpp).
enum cudaDeviceAttr {
    cudaDevAttrClockRate = 13,                       // no Vulkan limit for it: 0, and the caller prints it as unknown
    cudaDevAttrMultiProcessorCount = 16,             // VkPhysicalDeviceProperties::limits... maxComputeUnits
    cudaDevAttrIntegrated = 18,                      // deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    cudaDevAttrMemoryClockRate = 36,                 // no Vulkan limit for it either: 0 (same)
    cudaDevAttrGlobalMemoryBusWidth = 37,            //   (same)
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,    // VkPhysicalDeviceLimits::maxComputeLocalSize (Phase 0 P1 measured
                                                     //   both cards' workgroup-local sizes)
    cudaDevAttrComputeCapabilityMajor = 75,          // Vulkan has no compute capability: see the cudaDeviceProp note
    cudaDevAttrComputeCapabilityMinor = 76,
};

// cudaDeviceSchedule* / cudaDeviceMapHost: how a CUDA host thread waits for the GPU, and the flag that asks CUDA for
// mapped host memory (../src passes them to cudaSetDeviceFlags - src/core/overlaps.cpp / src/core/pinned.cu).  Vulkan
// waits on a timeline semaphore (vkWaitSemaphores) or on a submit; there is no spin/yield knob to set, so the flags are
// accepted and recorded rather than ignored in silence, and cudaDeviceMapHost is what this port always does: mapped
// host memory is host-visible host-coherent memory, and a shader is given its address.  CUDA spells all of these as
// flag values, not as functions, so cudaDeviceMapHost is a constant here and nothing calls a function of that name.
enum cudaDeviceScheduleFlags { cudaDeviceScheduleAuto = 0, cudaDeviceScheduleSpin = 1, cudaDeviceScheduleYield = 2,
                               cudaDeviceScheduleBlockingSync = 4, cudaDeviceScheduleLto = 8 };
inline constexpr unsigned cudaDeviceMapHost = 0x01;

// CUDA's toolkit version, and what it means where there is no CUDA.  ../src gates three things on it: the versioned
// driver-entry-point query (>= 12050), the CUDA 13 batched host->device copy (>= 13000; include/strata/core/
// dma_batch.hpp:68) and its 13.4 overlap hint (>= 13040), and src/program/generate.cpp's #542 check that the libcudart
// a build loaded matches the headers it was compiled with.  The value here is the CUDA 12.5 shape: the entry-point
// query IS provided (that is how src/core/vmm.cpp:25-53 and src/core/expert_cache.cpp:232-260 obtain the cuMem*
// functions, by name), and the CUDA 13 batch copy is NOT, because Vulkan has nothing to answer it with - dma_batch.hpp
// then takes its one-copy-per-blob loop, which is exactly the n vkCmdCopyBuffers in one command buffer the port would
// submit anyway.  cudaRuntimeGetVersion() reports this same number, so #542 cannot fire on a build with no libcudart.
#define CUDART_VERSION 12050

// ---- errors, devices ---------------------------------------------------------------------
inline cudaError_t cudaGetLastError() { return strata::gpu::get_last_error(); }
inline cudaError_t cudaPeekAtLastError() { return strata::gpu::peek_last_error(); }
inline const char* cudaGetErrorString(cudaError_t e) { return strata::gpu::error_string(e); }
inline const char* cudaGetErrorName(cudaError_t e) { return strata::gpu::error_name(e); }
inline cudaError_t cudaGetDevice(int* d) { return strata::gpu::get_device(d); }
inline cudaError_t cudaSetDevice(int d) { return strata::gpu::set_device(d); }
inline cudaError_t cudaInitDevice(int d, unsigned device_flags = 0, unsigned flags = 0) {
    // CUDA 12's cudaInitDevice(device, deviceFlags, flags): create the device's context now, with a scheduling policy
    // chosen before anything runs (src/core/remote_experts.cpp:80 asks for spin + mapped host memory so a CPU-side wait
    // on the second card does not pay a sleep/wake round trip).  Vulkan has no context to create lazily (the VkDevice
    // is opened when the port opens it) and no spin/yield policy to choose, so the flags are recorded, not obeyed.
    (void) device_flags; (void) flags;
    return strata::gpu::set_device(d);
}
inline cudaError_t cudaGetDeviceCount(int* n) { return strata::gpu::device_count(n); }
inline cudaError_t cudaDeviceSynchronize() { return strata::gpu::device_synchronize(); }
inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { return strata::gpu::mem_get_info(f, t); }   // VK_EXT_memory_budget
                                                                                                       // per card, minus this
                                                                                                       // process's own usage
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int d) { return strata::gpu::get_device_properties(p, d); }
inline cudaError_t cudaRuntimeGetVersion(int* v) { return strata::gpu::runtime_version(v); }
inline cudaError_t cudaDriverGetVersion(int* v) { return strata::gpu::runtime_version(v); }
inline cudaError_t cudaSetDeviceFlags(unsigned f) { return strata::gpu::device_set_flags(f); }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int d) { return strata::gpu::device_attr_int(v, (int) a, d); }
inline cudaError_t cudaDeviceCanAccessPeer(int* v, int d, int p) { return strata::gpu::device_can_access_peer(v, d, p); }
inline cudaError_t cudaDeviceEnablePeerAccess(int p, unsigned f) { return strata::gpu::device_enable_peer_access(p, f); }

// ---- memory and copies -------------------------------------------------------------------
// The two Arcs are two VkDevices, not two peer-accessible CUDA devices: there is no Vulkan peer-access to enable, and a
// cross-card transfer goes through the host, which is already how the engine hands a stage to the next card
// (src/core/remote_experts.cpp; the SYCL split's Verifier::forward_handoff).
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return strata::gpu::malloc_device(p, n); }
inline cudaError_t cudaFree(void* p) { return strata::gpu::free_device(p); }
template <class T> inline cudaError_t cudaHostAlloc(T** p, size_t n, unsigned f) { return strata::gpu::host_alloc(p, n, f); }
template <class T> inline cudaError_t cudaMallocHost(T** p, size_t n) { return strata::gpu::malloc_host(p, n); }
inline cudaError_t cudaFreeHost(void* p) { return strata::gpu::free_host(p); }
template <class T> inline cudaError_t cudaHostGetDevicePointer(T** d, void* h, unsigned f) {
    return strata::gpu::host_device_pointer(d, h, f);
}
inline cudaError_t cudaHostRegister(void* p, size_t n, unsigned f) { return strata::gpu::host_register(p, n, f); }
inline cudaError_t cudaHostUnregister(void* p) { return strata::gpu::host_unregister(p); }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { return strata::gpu::memset_device(p, v, n); }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t s = nullptr) {
    return strata::gpu::memset_async(p, v, n, s);   // vkCmdFillBuffer in the stream's command buffer
}
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind k) {
    return strata::gpu::memcpy_sync(d, s, n, k);
}
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t st = nullptr) {
    return strata::gpu::memcpy_async(d, s, n, k, st);
}
inline cudaError_t cudaMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, cudaMemcpyKind k,
                                     cudaStream_t st = nullptr) {
    return strata::gpu::memcpy_2d_async(d, dp, s, sp, w, h, k, st);   // one vkCmdCopyBuffer per row: Vulkan copies are
}                                                                     // contiguous on both sides
inline cudaError_t cudaMemcpyPeerAsync(void* d, int dd, const void* s, int sd, size_t n, cudaStream_t st = nullptr) {
    return strata::gpu::memcpy_peer_async(d, dd, s, sd, n, st);
}
// cudaMemcpyToSymbol: CUDA's copy into a __constant__ symbol.  This port has no constant symbols (a shader reads its
// inputs through buffers and push constants), so the call is only meaningful for a device buffer the caller already
// holds - which is what src/kernels/cuda/* uses it for on the CUDA side of the same names.
#define cudaMemcpyToSymbol(dst, src, n) cudaMemcpy((void*) (dst), (const void*) (src), (size_t) (n), cudaMemcpyHostToDevice)

// ---- streams, events, host functions -----------------------------------------------------
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { return strata::gpu::stream_create(s, 0); }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned f) { return strata::gpu::stream_create(s, f); }
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned f, int p) {
    return strata::gpu::stream_create(s, f);   // queue priority is a VkDeviceQueueCreateInfo priority in Vulkan, fixed
}                                             // when the device is opened: S3 answers the range it can honour
inline cudaError_t cudaDeviceGetStreamPriorityRange(int* low, int* high) { return strata::gpu::stream_priority_range(low, high); }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { return strata::gpu::stream_destroy(s); }
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) { return strata::gpu::stream_synchronize(s); }
inline cudaError_t cudaStreamQuery(cudaStream_t s) { return strata::gpu::stream_query(s); }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned f = 0) {
    return strata::gpu::stream_wait_event(s, e, f);   // vkWaitSemaphores for the event's timeline value, recorded as a
}                                                     // semaphore wait node in the stream's command buffer
inline cudaError_t cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* st) {
    int v = 0;
    const cudaError_t e = strata::gpu::stream_is_capturing(s, &v);
    if (st) *st = (cudaStreamCaptureStatus) v;
    return e;
}
inline cudaError_t cudaLaunchHostFunc(cudaStream_t s, cudaHostFn_t fn, void* a) { return strata::gpu::launch_host_func(s, fn, a); }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { return strata::gpu::event_create(e, 0); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned f) { return strata::gpu::event_create(e, f); }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { return strata::gpu::event_destroy(e); }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr) { return strata::gpu::event_record(e, s); }
inline cudaError_t cudaEventQuery(cudaEvent_t e) { return strata::gpu::event_query(e); }
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) { return strata::gpu::event_synchronize(e); }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    return strata::gpu::event_elapsed_ms(ms, a, b);   // the two timeline values, or a VK_KHR_calibrate_timestamps pair
}

// ---- graphs ------------------------------------------------------------------------------
// A verify window is one VkCommandBuffer recorded once (stream capture) and resubmitted per window; the engine keeps
// the per-window values in device buffers so a replay needs no graph update (VULKAN-BACKEND-PLAN.md's mapping table,
// and Phase 0 P4: 2400 dispatches in one buffer cost 0.14-0.16 us per dispatch on these cards).
inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode m) {
    // Only ThreadLocal is accepted, exactly as metal_compat does it: the engine records a window on its own stream
    // (src/core/verify.cpp:1891, :2026), and a Global capture would pull other threads' submissions into the same
    // command buffer.
    return m == cudaStreamCaptureModeThreadLocal ? strata::gpu::begin_capture(s) : strata::gpu::compat_detail::bad();
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* g) { return strata::gpu::end_capture(s, g); }
inline cudaError_t cudaGraphGetNodes(cudaGraph_t g, cudaGraphNode_t* nodes, size_t* n) {
    return strata::gpu::graph_nodes(g, nodes, n);
}

// cudaGraphNodeType / cudaGraphEdgeData / cudaKernelNodeParams: what verify.cpp's CUDA-only debug listing reads
// (src/core/verify.cpp:1947-2006 - node kinds, entry point names, and which edges are programmatic PDL edges).  The
// recording knows its nodes, so this port reports them; there is no programmatic edge on Vulkan (no PDL: the wait form
// that needs it is the doorbell the driver refused in Phase 0 P5), so every edge is an ordinary dependency and the
// listing says so instead of inventing one.
enum cudaGraphNodeType { cudaGraphNodeTypeInvalid = 0, cudaGraphNodeTypeKernel = 1, cudaGraphNodeTypeMemcpy = 2,
                         cudaGraphNodeTypeMemset = 3, cudaGraphNodeTypeHost = 4, cudaGraphNodeTypeGraph = 5,
                         cudaGraphNodeTypeEmpty = 6, cudaGraphNodeTypeWaitEvent = 7, cudaGraphNodeTypeEventRecord = 8,
                         cudaGraphNodeTypeExtSemaphoreSignal = 9, cudaGraphNodeTypeExtSemaphoreWait = 10 };
enum cudaGraphDependencyType { cudaGraphDependencyTypeDefault = 0, cudaGraphDependencyTypeProgrammatic = 1 };
struct cudaGraphEdgeData {
    unsigned char type;                       // cudaGraphDependencyType
    unsigned char unused_pad_[3];
    union { int pad; } attributes;
};
struct cudaKernelNodeParams {
    void* func;                               // the entry point (verify.cpp:1960-1963 reads only this)
    unsigned gridDim[3];
    unsigned blockDim[3];
    void** kernelParams;
    unsigned int extra;                       // cudaKernelNodeParams::extra in CUDA 12
    size_t sharedMemBytes;
};
inline cudaError_t cudaGraphNodeGetType(cudaGraphNode_t n, cudaGraphNodeType* type) {
    int t = 0;
    const cudaError_t e = strata::gpu::graph_node_type(n, &t);
    if (type) *type = (cudaGraphNodeType) t;
    return e;
}
inline cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t n, cudaKernelNodeParams* p) {
    if (p == nullptr) return cudaErrorInvalidValue;
    strata::gpu::Dim3 grid{}, block{};
    size_t shared = 0;
    const cudaError_t e = strata::gpu::graph_node_params(n, &p->func, &grid, &block, &shared);
    p->gridDim[0] = grid.x; p->gridDim[1] = grid.y; p->gridDim[2] = grid.z;
    p->blockDim[0] = block.x; p->blockDim[1] = block.y; p->blockDim[2] = block.z;
    p->sharedMemBytes = shared;
    return e;
}
inline cudaError_t cudaFuncGetName(const char** name, const void* func) { return strata::gpu::kernel_name(func, name); }
inline cudaError_t cudaGraphGetEdges(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to, cudaGraphEdgeData* ed,
                                     size_t* n) {
    int* types = nullptr;   // the edge kinds are read only through cudaGraphEdgeData::type; keep it simple and honest
    const cudaError_t e = strata::gpu::graph_edges(g, from, to, nullptr, n);
    (void) types;
    if (e == cudaSuccess && ed != nullptr && n != nullptr)
        for (size_t i = 0; i < *n; ++i) ed[i].type = (unsigned char) cudaGraphDependencyTypeDefault;
    return e;
}
inline cudaError_t cudaGraphGetEdges_v2(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to, cudaGraphEdgeData* ed,
                                        size_t* n) {
    return cudaGraphGetEdges(g, from, to, ed, n);   // CUDA 12.4 renamed nothing that matters here: same five arguments
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* x, cudaGraph_t g, cudaGraphNode_t*, char*, size_t) {
    return strata::gpu::graph_instantiate(x, g);
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* x, cudaGraph_t g, unsigned long long f = 0) {
    return strata::gpu::graph_instantiate(x, g, f);
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t x, cudaStream_t s) { return strata::gpu::graph_launch(x, s); }
inline cudaError_t cudaGraphUpload(cudaGraphExec_t x, cudaStream_t s) { return strata::gpu::graph_upload(x, s); }
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) { return strata::gpu::graph_destroy(g); }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t x) { return strata::gpu::graph_exec_destroy(x); }

// ---- CUDA's driver API, reached through the runtime --------------------------------------
// src/core/vmm.cpp:25-53 and src/core/expert_cache.cpp:232-260 do not link against libcuda: they ask the runtime for
// cuDeviceGetAttribute / cuMemCreate / cuMemMap ... by NAME through cudaGetDriverEntryPoint and call them through
// function pointers.  That is the hole this port has to fill honestly: there is no driver library here, so the same
// names are handed out as pointers to this port's own implementations (declared in vulkan_compat/cuda.h, defined in
// vulkan/src/gpu/runtime.cpp).  What the engine then does with them - one address range for the arena, physical chunks
// bound under it and unmapped again - is the segmented arena (--vram-elastic), and on Vulkan that is one
// VkDeviceMemory block set plus vkBindBufferMemory2 / view re-creation per chunk (Phase 1's sub-allocator).
enum cudaDriverEntryPointQueryResult { cudaDriverEntryPointSuccess = 0, cudaDriverEntryPointSymbolNotFound = 1,
                                       cudaDriverEntryPointVersionNotSufficed = 2 };
enum cudaDriverEntryPointFlags { cudaEnableDefault = 0x0, cudaEnableLegacy = 0x1, cudaEnablePerThreadDefault = 0x2 };
inline cudaError_t cudaGetDriverEntryPoint(const char* name, void** p, unsigned flags,
                                           cudaDriverEntryPointQueryResult* status) {
    (void) flags;
    strata::gpu::DriverFn fn = nullptr;
    const cudaError_t e = strata::gpu::driver_entry_point(name, &fn);
    if (status) *status = (e == cudaSuccess && fn != nullptr) ? cudaDriverEntryPointSuccess
                                                             : cudaDriverEntryPointSymbolNotFound;
    if (p) *p = fn;
    return e;
}
// The versioned spelling, which ../src takes when CUDART_VERSION is 12050 or newer (src/core/vmm.cpp:29,
// src/core/expert_cache.cpp:238: `cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q)`).  CUDA's
// third argument means "the driver must be at least this CUDA version"; there is no CUDA driver here to age, so the
// same table answers both spellings and anything this port does not implement comes back as SymbolNotFound - which is
// the answer vmm.cpp:33-36 already branches on (it gives up on the VMM path) and expert_cache.cpp:262 does the same.
inline cudaError_t cudaGetDriverEntryPointByVersion(const char* name, void** p, unsigned major_version, unsigned flags,
                                                    cudaDriverEntryPointQueryResult* status) {
    (void) major_version;
    return cudaGetDriverEntryPoint(name, p, flags, status);
}

// ---- the launch form ---------------------------------------------------------------------
// CUDA's `kernel<<<grid, block, sharedMem, stream>>>(args)` is not C++: only nvcc, hipcc and clang's CUDA mode tokenize
// <<< >>>, which is why include/strata/kernels/pdl.hpp keeps launch_pdl - whose last line IS that form (:95) - inside
// `#if defined(__CUDACC__) || defined(__HIPCC__)` (:34).  cl.exe has no <<< and nothing in the compiled host tree uses
// one, so the Vulkan port writes the same launch through this macro: the same four launch values in the same order,
// expanded into the typed host launch call that submits the SPIR-V entry point.  grid = the dispatch sizes, block = the
// workgroup size, sharedMem = the pipeline's local-size limit, stream = the command buffer the dispatch is recorded
// into, params = the kernel's arguments as the one parameter block a SPIR-V entry point takes (the multi-argument
// CUDA kernels already pass them as a struct: gr_down_kernel(GrMulti m), resolve_kernel(KvStreamMap m)).
// strata::gpu::launch_kernel / LaunchConfig are declared in vulkan/include/strata/gpu/runtime.hpp; S3 implements the
// dispatch over vkCmdDispatch with the parameter layout SPIR-V reflection describes.
#define strata_gpu_launch_config(grid, block, shared_mem, stream) \
    ::strata::gpu::LaunchConfig{::strata::gpu::Dim3{(grid).x, (grid).y, (grid).z}, \
                                ::strata::gpu::Dim3{(block).x, (block).y, (block).z}, (shared_mem), (stream)}
#define strata_launch_kernel(kernel, grid, block, shared_mem, stream, params) \
    ::strata::gpu::launch_kernel(#kernel, strata_gpu_launch_config((grid), (block), (shared_mem), (stream)), \
                                 &(params), sizeof(params))

// ---- how this inventory got here: the error-first loop -----------------------------------
// Transcripts: build-vulkan\logs\iter-00.log ... iter-09.log (one build per iteration; the counts below are the
// `: error ` lines cl.exe wrote in that run).  The descent, in the same order the fixes were made:
//   iter 0  4356 error lines.  This file was `#pragma once` and nothing else, over 144 host TUs of ../src:
//           1732 C3861 ("cudaMalloc": identifier not found), 1624 C2065 ('cudaStream_t': undeclared identifier),
//           187 C2146 / 136 C2143 / 90 C2238 / 87 C3646 / 82 C2059 / 76 C4430 (the syntax fallout of those), 36 C2039,
//           21 C1003 (cl.exe stops a TU at 100 errors, so the true count is higher), 7 C1083 naming headers the tree
//           asks for - 'cuda.h' (src/core/vmm.cpp:4, src/core/expert_cache.cpp:12), 'cuda_fp16.h'
//           (src/kernels/qsa_prompt_attn_parity.cpp:17), 'unistd.h' (src/core/pinned_shared_test.cpp:9),
//           'sycl/sycl.hpp' (include/strata/core/native_head.hpp:10) - and 2 C1189 from the CPU kernel files that
//           #error without their /arch: flag.  71 distinct cuda*/CU* identifiers were named outright, and those 71
//           are what this header was written from (cross-checked against a scan of the same names in
//           include/strata/**/*.hpp, which the error cap had hidden).
//   iter 1  19 error lines: this file written, vulkan_compat/cuda.h, cuda_fp16.h, sycl/sycl.hpp, the /arch: flags,
//           vulkan/src/gpu/runtime.cpp and vulkan/src/core/device_vk.cpp.  All 19 were this header's own mistakes
//           (a Dim3 declared twice, cudaDeviceMapHost spelled as both a flag and a function).
//   iter 2  4 error lines: cudaGetDriverEntryPointByVersion not found (vmm.cpp:29, expert_cache.cpp:238 - the
//           CUDART_VERSION >= 12050 branch), cudaInitDevice called with CUDA's three arguments
//           (remote_experts.cpp:80), and cudaDeviceProp::luid missing (pinned.cu:368).
//   iter 3  1 error line: p.luid is CUDA's `char luid[8]`, not a struct - pinned.cu passes it as the const void*
//           include/strata/platform/memory.hpp:35 promises ("8 bytes at `luid`").
//   iter 4  0 compile errors, and the link became readable: 1 LNK2005 (main already defined in generate.obj) and 102
//           LNK4006 from the harness TUs' own main()s, plus the 406 unresolved-external lines this state leaves.
//   iter 5  4 error lines, all C1083, from trying to compile as C++ the four other .cu files that have no <<< in them:
//           'common.cuh' (ggml's CUDA backend internal, from ggml_cuda_host.cu:4, mmq_kq_lib.cu:8, mmq_nvfp4_w4a8.cu:10)
//           and 'cublas_v2.h' (gemm_bf16_parity.cu:13).  Those four are CUDA host code in the other sense and stay out.
//   iter 6/7  0 compile errors; 101 warning LNK4006 lines from LIB's own duplicate-main report on the harness archive.
//   iter 8  0 compile errors, 0 LNK4006, and the tail is what Phase 1 wants: 278 error LNK2019 + 128 error LNK2001
//           lines over 280 symbols, every one of them defined in one of the 63 ../src .cu files this build does not
//           compile, and 1 fatal LNK1120.  Nothing else fails.
//
// What is mapped for what reason, because "everything CUDA has" is not the method.  Checked name by name against the
// compiled host set (all 143 ../src .cpp + src/core/pinned.cu + the include/strata headers they pull):
//   named by the compiled tree          cudaGetLastError / cudaPeekAtLastError / cudaGetErrorString, cudaGetDevice /
//                                      cudaSetDevice / cudaGetDeviceCount / cudaDeviceSynchronize / cudaMemGetInfo
//                                      (src/core/expert_cache.cpp:26) / cudaGetDeviceProperties / cudaDeviceGetAttribute
//                                      / cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess, the cudaError* values
//                                      ../src compares against (src/core/conversation_validation_test.cpp:22,37),
//                                      cudaMalloc / cudaFree / cudaHostAlloc (expert_cache.cpp:451) / cudaMallocHost
//                                      (src/program/generate.cpp:1667) / cudaFreeHost / cudaHostGetDevicePointer
//                                      (src/core/expert_source.cpp:2901) / cudaHostRegister (:2992) / cudaHostUnregister
//                                      (:2837) / cudaMemset / cudaMemsetAsync (src/core/layer.cpp:221) / cudaMemcpy /
//                                      cudaMemcpyAsync / cudaMemcpy2DAsync (src/core/layer.cpp:1117) / cudaMemcpyPeerAsync
//                                      (src/prefill/prefill.cpp:645), the stream, event and graph calls (mtp.cpp:440-441
//                                      for the priority pair, generate.cpp:6409 for cudaStreamLegacy, prefill.cpp:465 for
//                                      cudaEventBlockingSync, verify.cpp:1956-1992 for the node/edge listing),
//                                      cudaInitDevice + cudaDeviceScheduleSpin + cudaDeviceMapHost (all three on
//                                      remote_experts.cpp:80), CUDART_VERSION (expert_cache.cpp:237, verify.cpp:1961),
//                                      cudaGetDriverEntryPointByVersion (vmm.cpp:29, expert_cache.cpp:238).
//   carried forward, named by the same tree's CUDA-only files   cudaMemcpyToSymbol (src/kernels/cuda/s2_gemv_fast.cu:59,
//                                      src/prefill/kernels.cu:1824) and cudaStreamIsCapturing (src/kernels/cuda/
//                                      sampler.cu:1128): no compiled host file uses them, the .cu files S4 ports do, and
//                                      a compat header that waits for the next phase to ask is a header that gets
//                                      rewritten every phase.
//   carried from the HIP build's proven inventory of this same tree   cudaGetErrorName, cudaSetDeviceFlags,
//                                      cudaHostAllocWriteCombined, cudaHostRegisterDefault - the only places they
//                                      appear are include/strata/platform/hip_compat/strata_hip.h:31,57,58,81,102, where
//                                      the HIP port already had to answer them.
// cudaErrorStreamCaptureUnsupported is not forward-looking: include/strata/kernels/shared_expert.hpp:46 uses it in a
// header the host tree compiles.  cudaErrorLaunchFailure is not a name ../src asks for; it is a value of this port's
// own Error enum, used by error_string()/error_name() to name what a failed vkQueueSubmit means.

// What the loop did NOT ask for, and why that is worth writing down: <cpuid.h> never produced a C1083, because
// src/kernels/cpu/expert.cpp:8-12 and expert_layout.cpp:9-13 already take `#if defined(_MSC_VER) #include <intrin.h>`
// (metal_compat/cpuid.h exists because arm64 has no cpuid at all; here the real __cpuidex runs).  <immintrin.h> needs
// no shim either: MSVC's own is the right one for an x86 PC.  <unistd.h> is asked for by exactly one TU, the Linux
// MAP_SHARED arena selftest, which upstream builds only on Linux (see vulkan/CMakeLists.txt) - mapping mkstemp and
// /tmp onto Windows to make that "compile" would be a shim that lies.

