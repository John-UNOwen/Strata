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
// The rounding to BF16 happens in the SPIR-V by VALUE (see vulkan/shaders/elementwise_dequant_bf16.comp): GLSL has no
// float-to-bits reinterpret, and the first attempt at that - binding one allocation as floats in one binding and as
// uints in another, then reading back what was written - produced values that were neither the products nor zero.
// This route allocates nothing but the output the caller passed.
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

// The .comp's push block, field for field.  Words 0..1 are the storage bindings in that order (blocks, then the
// output), because record_dispatch binds word i at binding i.
struct DeqArgs {
    uint64_t blocks;
    uint64_t out;
    int64_t total_blocks;
    int64_t block_first_byte;
    int32_t block_bytes;
    int32_t block_elems;
    int32_t pad0;
    int32_t pad1;
    uint64_t stream;
};

static_assert(sizeof(DeqArgs) == 56, "DeqArgs must match elementwise_dequant_bf16.comp's push block");

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
    spec.param_bytes = sizeof(DeqArgs);              // 56 B, well under this card's 256 B push limit
    spec.storage_bindings = 2;                       // blocks, then the output
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
    const DeqArgs args{
        (uint64_t) (uintptr_t) blocks,
        (uint64_t) (uintptr_t) out,
        total_blocks,
        row0 * blocks_per_row * kQ8Bytes,
        (int32_t) kQ8Bytes,
        (int32_t) kQ8Elems,
        0,                                             // pad0, the .comp's field for field
        0,                                             // pad1
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
