// vulkan/probe/vk_norm_probe.cpp - a small fixed-input run for the two elementwise cases
// src/kernels/elementwise_parity.cpp reports as wrong: rms_norm_weighted and dequant_bf16 (Q8_0).
//
// The shared driver says "rms_norm_weighted vs f64 oracle rel 7.652e-01 worst 1.693e+00" and "dequant_bf16 64 of 64
// wrong".  Those fixtures are big (24 x 256) and their numbers do not say WHICH part moved.  This probe uses cases
// small enough to print every element, with an oracle written here from the header's own definition, so a wrong row
// mapping, a wrong weight index, a sum read as a mean, or a scale read at the wrong byte shows up as a specific
// mismatch rather than a percentage.
//
// Case 1, rms: rows = 2, cols = 32.  Row 0 is 1..32, row 1 is all 2.  Weights are 1 + c/100, so a weight applied to
// the wrong column is visible per column, and the two rows have clearly different means, so a row mix-up is visible.
// Case 2, rms: rows = 24, cols = 256, the driver's shape, reported as totals only.
// Case 3, dequant: four Q8_0 blocks, scales 1.0, 2.0, a quiet NaN (the driver's row at :442) and a +infinity.  The
// NaN row and the -50..43 row both contain a zero weight, so NaN * 0 and inf * 0 are reached as well.
#include <cuda_runtime.h>

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata;

namespace {

int g_fail = 0;

uint16_t bf16_of(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    if ((i & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t) ((i >> 16) | 64u);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}

void check_norm(int64_t rows, int64_t cols, bool with_w, bool verbose) {
    std::vector<float> x((size_t) (rows * cols));
    std::vector<float> w((size_t) cols, 1.0f);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            x[(size_t) (r * cols + c)] = (rows == 2 && r == 1) ? 2.0f : (float) (c + 1);
    for (int64_t c = 0; c < cols; ++c) w[(size_t) c] = 1.0f + (float) c / 100.0f;

    float* d_x = nullptr;
    float* d_w = nullptr;
    if (cudaMalloc(&d_x, x.size() * 4) != cudaSuccess) { std::printf("  norm probe: cudaMalloc x failed\n"); ++g_fail; return; }
    if (cudaMalloc(&d_w, w.size() * 4) != cudaSuccess) { std::printf("  norm probe: cudaMalloc w failed\n"); ++g_fail; cudaFree(d_x); return; }
    cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice);

    const float eps = 1e-5f;
    kernels::rms_norm_weighted(d_x, with_w ? d_w : nullptr, rows, cols, eps, nullptr);
    std::vector<float> got((size_t) (rows * cols), 0.0f);
    cudaMemcpy(got.data(), d_x, got.size() * 4, cudaMemcpyDeviceToHost);

    std::vector<float> want((size_t) (rows * cols), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int64_t c = 0; c < cols; ++c) { const double v = x[(size_t) (r * cols + c)]; acc += v * v; }
        const double inv = 1.0 / std::sqrt(acc / (double) cols + (double) eps);
        for (int64_t c = 0; c < cols; ++c) {
            const double v = x[(size_t) (r * cols + c)];
            want[(size_t) (r * cols + c)] = (float) (v * inv * (with_w ? w[(size_t) c] : 1.0));
        }
    }

    int bad = 0;
    double worst = 0.0;
    int64_t worst_at = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        const double m = std::fabs((double) want[i]);
        const double rel = std::fabs((double) got[i] - (double) want[i]) / (m > 1e-30 ? m : 1e-30);
        if (rel > 1e-5) ++bad;
        if (rel > worst) { worst = rel; worst_at = (int64_t) i; }
    }
    std::printf("  rms rows %lld cols %lld weights %s : %d of %zu differ (tol 1e-5), worst rel %.3e at index %lld\n",
                (long long) rows, (long long) cols, with_w ? "yes" : "null", bad, got.size(), worst,
                (long long) (worst_at < 0 ? -1 : worst_at % cols));
    if (verbose) {
        for (int64_t r = 0; r < 2 && r < rows; ++r) {
            for (int64_t c = 0; c < 8 && c < cols; ++c) {
                const size_t i = (size_t) (r * cols + c);
                std::printf("    r%lld c%lld  want %12.6f  got %12.6f\n", (long long) r, (long long) c, want[i], got[i]);
            }
        }
        // The l2 norm of each output row is the fastest way to see a sum read as a mean, or one row's mean used for
        // every row: the reference rows have norms that differ by a factor of 16.
        for (int64_t r = 0; r < rows; ++r) {
            double g = 0.0, h = 0.0;
            for (int64_t c = 0; c < cols; ++c) {
                g += (double) got[(size_t) (r * cols + c)] * got[(size_t) (r * cols + c)];
                h += (double) want[(size_t) (r * cols + c)] * want[(size_t) (r * cols + c)];
            }
            std::printf("    row %lld l2: want %.6f got %.6f\n", (long long) r, std::sqrt(h), std::sqrt(g));
        }
    }
    if (bad) ++g_fail;
    cudaFree(d_x);
    cudaFree(d_w);
}

void check_dequant() {
    // Q8_0 is a 2-byte fp16 scale FIRST and then 32 int8 weights, 34 bytes per block, which is what
    // src/kernels/elementwise_parity.cpp:441 writes and what the first version of the shader read backwards.
    const int kBlocks = 4;
    std::vector<uint8_t> blk((size_t) (kBlocks * 34), 0);
    blk[0] = 0x00; blk[1] = 0x3C;                            // fp16 1.0
    blk[34] = 0x00; blk[35] = 0x40;                          // fp16 2.0
    blk[68] = 0x00; blk[69] = 0x7E;                          // fp16 quiet NaN, the driver's row 0
    blk[102] = 0x00; blk[103] = 0x7C;                        // fp16 +infinity
    for (int j = 0; j < 32; ++j) {
        blk[(size_t) (2 + j)] = (uint8_t) (int8_t) (j + 1);          // 1 to 32
        blk[(size_t) (36 + j)] = (uint8_t) (int8_t) (j - 15);        // -15 to 16
        blk[(size_t) (70 + j)] = (uint8_t) (int8_t) (j - 16);        // -16 to 15, has a zero
        blk[(size_t) (104 + j)] = (uint8_t) (int8_t) (3 * j - 45);   // -45 to 48, has a zero at j = 15
    }

    uint8_t* d_b = nullptr;
    uint16_t* d_o = nullptr;
    if (cudaMalloc(&d_b, blk.size()) != cudaSuccess) { std::printf("  dequant probe: cudaMalloc blocks failed\n"); ++g_fail; return; }
    if (cudaMalloc(&d_o, kBlocks * 32 * 2) != cudaSuccess) { std::printf("  dequant probe: cudaMalloc out failed\n"); ++g_fail; cudaFree(d_b); return; }
    cudaMemcpy(d_b, blk.data(), blk.size(), cudaMemcpyHostToDevice);
    std::printf("  dequant_bf16_supported(8) says %s\n", kernels::dequant_bf16_supported(8) ? "true" : "false");
    kernels::dequant_bf16(8, d_b, 0, kBlocks, 32, d_o, nullptr);

    std::vector<uint16_t> got((size_t) (kBlocks * 32), 0);
    cudaMemcpy(got.data(), d_o, got.size() * 2, cudaMemcpyDeviceToHost);
    // Read the blocks back too: if the upload did not land, or landed somewhere else, the kernel is not the suspect.
    std::vector<uint8_t> back(blk.size(), 0);
    const cudaError_t back_err = cudaMemcpy(back.data(), d_b, back.size(), cudaMemcpyHostToDevice);
    std::printf("    blocks read back (%s), scale bytes:", cudaGetErrorString(back_err));
    for (int b = 0; b < kBlocks; ++b) std::printf(" %02x%02x", back[(size_t) (b * 34)], back[(size_t) (b * 34 + 1)]);
    std::printf("\n");

    int bad = 0;
    for (int j = 0; j < 32; ++j) {
        if (got[(size_t) j] != bf16_of((float) (j + 1))) ++bad;                            // d = 1.0
        if (got[(size_t) (32 + j)] != bf16_of(2.0f * (float) (j - 15))) ++bad;            // d = 2.0
        if (!((got[(size_t) (64 + j)] & 0x7FFFu) > 0x7F80u)) ++bad;                       // d = NaN, any sign
        const int q = 3 * j - 45;
        const uint16_t want_inf = (uint16_t) ((q < 0 ? 0x8000u : 0u) | (q == 0 ? 0x7FC0u : 0x7F80u));
        const uint16_t h = got[(size_t) (96 + j)];
        const bool ok = (q == 0) ? ((h & 0x7FFFu) > 0x7F80u) : (h == want_inf);            // inf * 0 is a NaN
        if (!ok) ++bad;
    }
    std::printf("    b0 q 1 want %04x got %04x ; b1 q -15 want %04x got %04x ; b2 q 0 with a NaN scale got %04x ; "
                "b3 q -45 want %04x got %04x\n",
                bf16_of(1.0f), got[0], bf16_of(-30.0f), got[32], got[80], 0xFF80u, got[96]);
    std::printf("  dequant_bf16 Q8_0, %d blocks with a NaN and an infinity scale : %d value(s) wrong\n", kBlocks, bad);
    if (bad) ++g_fail;
    cudaFree(d_b);
    cudaFree(d_o);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);        // a run killed by a timeout must still show what it printed
    std::printf("vk_norm_probe\n");
    check_norm(2, 32, true, true);
    check_norm(2, 32, false, false);
    check_norm(24, 256, true, false);
    check_dequant();
    std::printf("vk_norm_probe: %d failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
