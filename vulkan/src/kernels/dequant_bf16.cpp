// vulkan/src/kernels/dequant_bf16.cpp - strata::kernels::dequant_bf16 and dequant_bf16_supported, Q8_0 only.
//
// include/strata/kernels/dequant_bf16.hpp declares the family (Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
// IQ4_XS, Q2_0) and src/kernels/cuda/dequant_bf16.cu:308 is the geometry table it transcribes.  This file ports
// Q8_0 (ggml type 8: 32 signed bytes plus an fp16 scale, 34 bytes per block) and says so: dequant_bf16_supported
// reports true for type 8 and false for the other nine, and dequant_bf16 refuses a type it has no body for instead
// of writing something that merely looks like a dequantized row.  The nine that are not here yet are Phase 3 work,
// the prompt path's quantized-weight GEMMs; src/kernels/elementwise_parity.cpp reaches this one because its last
// case is a Q8_0 block whose scale is a NaN.
//
// The scratch buffer is the shader's bitcast (see vulkan/shaders/elementwise_dequant_bf16.comp): the products have
// to be read back as bits to be rounded to BF16, so the kernel needs rows*cols floats of device memory in addition
// to the output.  It is kept for the life of the process and grown when a bigger call comes, rather than allocated
// and freed per launch: a free right after a launch on a caller's stream would hand back memory the GPU is still
// reading.  Slicing a large tensor into chunks, so the scratch stays bounded, is the Phase 3 follow-up.
#include "strata/kernels/dequant_bf16.hpp"

#include "strata/gpu/runtime.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__has_include)
#  if __has_include("vulkan_shaders/elementwise_dequant_bf16.h")
#    include "vulkan_shaders/elementwise_dequant_bf16.h"
#    define STRATA_VULKAN_HAS_DEQUANT_SPV 1
#  endif
#endif

namespace strata::kernels {
namespace {

constexpr const char* kLaunchName = "dequant_bf16";
constexpr unsigned kThreads = 256;                  // elementwise.cu's THREADS, and this .comp's local_size_x
constexpr int kQ8Elems = 32;                        // geometry(8): 32 elements
constexpr int kQ8Bytes = 34;                        // geometry(8): 34 bytes per block

// The .comp's push block, field for field.  Words 0..3 are the storage bindings in that order: blocks, the scratch
// seen as floats, the same scratch seen as uints, and the output.  The scratch is bound twice because GLSL cannot
// reinterpret a float as its bits, and record_dispatch binds word i at binding i.
struct DeqArgs {
    uint64_t blocks;
    uint64_t scratch;
    uint64_t scratch_bits;
    uint64_t out;
    int64_t blocks_per_row;
    int64_t total_blocks;
    int64_t block_first_byte;
    int32_t block_bytes;
    int32_t block_elems;
    uint64_t stream;
};

static_assert(sizeof(DeqArgs) == 72, "DeqArgs must match elementwise_dequant_bf16.comp's push block");

// The process-wide scratch, grown on demand.  0 while never allocated.
uint64_t g_scratch_addr = 0;
int64_t g_scratch_bytes = 0;

bool scratch_ready(int64_t bytes) {
    if (bytes <= g_scratch_bytes) return true;
    if (g_scratch_addr != 0) {
        const cudaError_t f = cudaFree((void*) (uintptr_t) g_scratch_addr);
        if (f != cudaSuccess) {
            std::fprintf(stderr, "strata vulkan: dequant_bf16: cudaFree of the %lld byte scratch said %s.  The old "
                                 "scratch stays in place and this call is refused.\n", (long long) g_scratch_bytes,
                         cudaGetErrorString(f));
            return false;
        }
        g_scratch_addr = 0;
        g_scratch_bytes = 0;
    }
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, (size_t) bytes);
    if (e != cudaSuccess || p == nullptr) {
        std::fprintf(stderr, "strata vulkan: dequant_bf16: cudaMalloc of %lld scratch bytes said %s.  The scratch is "
                             "rows x cols floats, so this is the dequant of a big tensor needing a chunked call.\n",
                     (long long) bytes, cudaGetErrorString(e));
        return false;
    }
    g_scratch_addr = (uint64_t) (uintptr_t) p;
    g_scratch_bytes = bytes;
    return true;
}

int g_pipeline_state = 0;                            // 0 never tried, 1 registered, 2 refused

bool pipeline_registered() {
    if (g_pipeline_state == 1) return true;
    if (g_pipeline_state == 2) return false;
    g_pipeline_state = 2;
#ifndef STRATA_VULKAN_HAS_DEQUANT_SPV
    std::fprintf(stderr, "strata vulkan: dequant_bf16: no SPIR-V to register: generated/vulkan_shaders/"
                         "elementwise_dequant_bf16.h was not built (vulkan/shaders/elementwise_dequant_bf16.comp is "
                         "not in STRATA_VULKAN_SHADERS in vulkan/CMakeLists.txt, or glslc failed).  Nothing is "
                         "dispatched and nothing is pretended.\n");
    return false;
#else
    static const std::vector<uint32_t> words = [] {
        std::vector<uint32_t> w((elementwise_dequant_bf16_spv_len + 3) / 4, 0u);
        std::memcpy(w.data(), elementwise_dequant_bf16_spv, elementwise_dequant_bf16_spv_len);
        return w;
    }();
    gpu::PipelineSpec spec;
    spec.name = kLaunchName;
    spec.entry = "main";
    spec.spv = words.data();
    spec.spv_words = words.size();
    spec.workgroup = gpu::Dim3{kThreads, 1, 1};
    spec.shared_mem = 0;
    spec.param_bytes = sizeof(DeqArgs);              // 72 B, well under this card's 256 B push limit
    spec.storage_bindings = 4;                       // blocks, scratch as floats, scratch as uints, out
    const gpu::Error e = gpu::register_pipeline(spec);
    if (e != gpu::kSuccess) {
        std::fprintf(stderr, "strata vulkan: dequant_bf16: register_pipeline(\"%s\") said %s (%s).  This entry point "
                             "dispatches nothing.\n", kLaunchName, gpu::error_name(e), gpu::error_string(e));
        return false;
    }
    g_pipeline_state = 1;
    return true;
#endif
}

}  // namespace

bool dequant_bf16_supported(int ggml_type) noexcept { return ggml_type == 8; }   // Q8_0; the other nine are not here

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    if (ggml_type != 8) {
        std::fprintf(stderr, "strata vulkan: dequant_bf16: refused ggml type %d - this backend has a Q8_0 body and "
                             "nothing else (see vulkan/shaders/elementwise_dequant_bf16.comp).  No row was written "
                             "and dequant_bf16_supported(%d) says false.\n", ggml_type, ggml_type);
        return;
    }
    if (rows <= 0 || cols <= 0 || cols % kQ8Elems != 0) {
        std::fprintf(stderr, "strata vulkan: dequant_bf16: refused shape %lld x %lld for type 8 - cols must be a "
                             "multiple of %d.  No row was written.\n", (long long) rows, (long long) cols, kQ8Elems);
        return;
    }
    if (blocks == nullptr || out == nullptr || row0 < 0) {
        std::fprintf(stderr, "strata vulkan: dequant_bf16: refused - blocks %p, out %p, row0 %lld.\n", (void*) blocks,
                     (void*) out, (long long) row0);
        return;
    }
    if (!pipeline_registered()) return;                       // the reason is already on stderr

    const int64_t blocks_per_row = cols / kQ8Elems;
    const int64_t total_blocks = rows * blocks_per_row;
    if (!scratch_ready(total_blocks * kQ8Elems * 4)) return;  // the reason is already on stderr

    const DeqArgs args{
        (uint64_t) (uintptr_t) blocks,
        g_scratch_addr,
        g_scratch_addr,                                    // the same allocation, the other view of it
        (uint64_t) (uintptr_t) out,
        blocks_per_row,
        total_blocks,
        row0 * blocks_per_row * kQ8Bytes,
        (int32_t) kQ8Bytes,
        (int32_t) kQ8Elems,
        (uint64_t) (uintptr_t) stream,
    };

    gpu::LaunchConfig cfg;
    cfg.grid = gpu::Dim3{(unsigned) ((total_blocks + kThreads - 1) / kThreads), 1, 1};
    cfg.block = gpu::Dim3{kThreads, 1, 1};
    cfg.shared_mem = 0;
    cfg.stream = static_cast<gpu::Stream>(stream);

    const gpu::Error e = gpu::launch_kernel(kLaunchName, cfg, &args, sizeof args);
    if (e != gpu::kSuccess) {
        std::fprintf(stderr, "strata vulkan: dequant_bf16: launch_kernel over %lld blocks said %s (%s).  Nothing was "
                             "written.\n", (long long) total_blocks, gpu::error_name(e), gpu::error_string(e));
        return;
    }
    if (stream == nullptr) {                                 // the elementwise.cu:113-120 rule
        const gpu::Error s = gpu::device_synchronize();
        if (s != gpu::kSuccess) {
            std::fprintf(stderr, "strata vulkan: dequant_bf16: device_synchronize after the launch said %s (%s).\n",
                         gpu::error_name(s), gpu::error_string(s));
        }
    }
}

}  // namespace strata::kernels
