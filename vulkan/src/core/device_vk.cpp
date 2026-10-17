// vulkan/src/core/device_vk.cpp - strata::core's device facts and device arena, for the Vulkan build.
//
// The shared host tree calls these (include/strata/core/device.hpp) and src/core/device.cu is what implements them on
// CUDA and HIP.  That file cannot be compiled here: its poison path is CUDA-shaped - `#if !defined(STRATA_GPU_METAL)`
// at :22 opens a `__global__ void poison_kernel`, `cudaFuncGetAttributes(&a, poison_kernel)` at :242, and
// `poison_kernel<<<(unsigned) chunk, threads>>>((float*) base_ + b * threads, n - b * threads)` at :342, and cl.exe has
// no <<< at all (only nvcc / hipcc / clang's CUDA mode tokenize it).  So the Vulkan port supplies the same symbols as
// its own file, over the compat names in vulkan/include/vulkan_compat - the same position Metal is in
// (metal/src/... has no device.cu either, and the Metal build's poison goes through strata::gpu::host_view, which is
// device.cu's own `#if defined(STRATA_GPU_METAL)` branch at :327-331).
//
// What is genuinely different on Vulkan, stated rather than hidden:
//   * There is no compute capability.  VkPhysicalDeviceProperties has no such field, and cudaDeviceProp's major/minor
//     here carry the device's Vulkan API version instead (see vulkan/include/strata/gpu/runtime.hpp's DeviceProp).
//     device.cu's runtime gate "compute capability 7.5 or newer" (:311) is therefore not a question this port can
//     answer; the Vulkan requirement is the feature set Phase 0 measured (API 1.3 core with timeline semaphores,
//     VK_EXT_external_memory_host, VK_KHR_driver_properties) and that is what device_info checks below.
//   * There is no "this binary was compiled for another GPU" failure.  gpu_arch_problem() and device_code_error()
//     exist to catch a cubin/HSIF built for sm_90 running on sm_120, or a gfx1100 .so on a gfx1201 card - the failure
//     device.cu detects with cudaFuncGetAttributes on a kernel that is not in the binary (:241-245).  A SPIR-V module
//     is not architecture-bound like that, so both return "" (which is what device.cu's own HIP/Metal branch at :237-238
//     does for the same reason) and a real failure would show up as a pipeline-creation error, reported per device.
#include "strata/core/device.hpp"

#include <cuda_runtime.h>   // vulkan_compat: the same names device.cu uses, forwarding to strata::gpu

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {
namespace {

[[noreturn]] void fail(const std::string& what, int code) { throw CudaError(what, code); }

void check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    fail(std::string(what) + ": " + cudaGetErrorString(e), (int) e);
}

std::string gib(size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2f GiB", (double) bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

}  // namespace

// CUDA/HIP answer the question "which GPU architectures is the device code in this binary for" (STRATA_HIP_ARCHS,
// CMAKE_CUDA_ARCHITECTURES).  A SPIR-V module this port builds at run time has no such list, so there is nothing to
// report - which is also what the CUDA branch of device.cu:70-76 returns for a CUDA build when nothing is named.
const char* compiled_gpu_archs() { return ""; }

// device.cu:217-234: on HIP, a card whose gcnArchName is not in the compiled list would fail at its first launch.
// On Vulkan a SPIR-V pipeline is validated when it is created, per VkDevice, and Phase 0 created pipelines on both
// cards of this PC (bench/results/vulkan-caps-2026-10-17.md), so an architecture mismatch is not a failure mode here.
std::string gpu_arch_problem(int ordinal) { (void) ordinal; return {}; }

// device.cu:236-247 does this by asking CUDA for the attributes of a kernel in this binary.  There is no kernel in
// this binary to ask about; the equivalent question - "can this device run the pipelines this port builds" - is
// answered by pipeline creation on that device, and a failure there is reported by the call that fails.
std::string device_code_error() { return {}; }

int device_count() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;   // 0 when there is no usable runtime, as device.cu:78-86
}

// One line per device for `strata-device --list-devices`.  device.cu:87-135 prints "compute capability 12.0, 11.9 GiB"
// on CUDA and "arch gfx1201, 15.9 GiB" on HIP; the Vulkan sentence names what Vulkan actually reports for the card.
bool device_summary(int ordinal, std::string& name, std::string& detail) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) return false;   // the runtime cannot describe it yet
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) total_b = p.totalGlobalMem;
    name = p.name;
    detail = "Vulkan API " + std::to_string(p.major) + "." + std::to_string(p.minor) + ", " + gib(total_b) +
             ", " + std::to_string(p.multiProcessorCount) + " compute units";
    return true;
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0)
        fail("no Vulkan device is present; Strata's Vulkan backend needs a Vulkan 1.3 device with "
             "VK_EXT_external_memory_host (Phase 0: bench/results/vulkan-caps-2026-10-17.md)",
             -1);
    if (ordinal < 0 || ordinal >= count)
        fail("device ordinal " + std::to_string(ordinal) + " is out of range (have " + std::to_string(count) + ")", -1);

    DeviceInfo d;
    d.ordinal = ordinal;
    check(cudaSetDevice(ordinal), "cudaSetDevice");

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    d.name = p.name;
    d.cc_major = p.major;                    // Vulkan: the API version, not a compute capability - see the note above
    d.cc_minor = p.minor;
    d.multi_processor_count = p.multiProcessorCount;
    d.arch.clear();                          // no gcnArchName-style arch string on Vulkan

    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    check(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&d.runtime_version), "cudaRuntimeGetVersion");

    // device.cu's gate (:311: "compute capability 7.5 or newer") has no Vulkan reading; the Vulkan requirement this
    // port actually depends on is the API version its own calls are written against (1.3: timeline semaphores are core
    // there), so that is what is checked.  A card below it is refused here rather than at the first dispatch.
    if (p.major < 1 || (p.major == 1 && p.minor < 3))
        fail("device " + d.name + " reports Vulkan API " + std::to_string(p.major) + "." + std::to_string(p.minor) +
                 "; the Vulkan backend needs Vulkan 1.3 (timeline semaphores) and VK_EXT_external_memory_host",
             -1);
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) fail("DeviceArena of 0 bytes", -1);
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    // One allocation for the whole region, before anything depends on it (device.cu:320-326 says why).
    check(cudaMalloc(&base_, (size_t) bytes), "cudaMalloc");
    if (poison_) {
        // The NaN-ish pattern, written the way the non-CUDA branch of device.cu writes it (:327-331): through the
        // allocation's CPU view.  There is no kernel to launch here, and there must not be a fake one.
        uint32_t* h = (uint32_t*) strata::gpu::host_view(base_);
        if (h == nullptr)
            fail("DeviceArena poison: this build has no host view of the device allocation yet "
                 "(strata::gpu::host_view, Phase 1 S3)",
                 -1);
        for (uint64_t i = 0; i < bytes / sizeof(uint32_t); ++i) h[i] = 0x7fc00000u;   // a quiet NaN per float
    }
}

DeviceArena::~DeviceArena() {
    if (base_) cudaFree(base_);   // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) fail("DeviceArena::alloc alignment must be a power of two", -1);
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B region - "
                      "the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        fail(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
