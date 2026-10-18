// vulkan/probe/vk_capture_probe.cpp - the graph-capture path, with no compute kernel in it.
//
// strata_parity_elementwise.exe stopped producing output inside src/kernels/elementwise_parity.cpp:382-392 (capture,
// instantiate, cudaGraphLaunch, cudaMemcpyAsync, cudaStreamSynchronize) right after two refused dispatches, and the
// Intel driver then had to be restarted.  That block had run fine earlier the same day with the same refusals, so the
// question is what the capture path does when the capture recorded NOTHING: end_capture closes a VkCommandBuffer with
// zero commands, graph_launch submits it, and the sync waits on a timeline value that an empty submit may never
// advance.  Each case here prints its own step before making it, unbuffered, so a killed process still says where it
// stopped.  No kernel is dispatched in cases A and B, so nothing here can wedge a workgroup.
//
// A: capture nothing at all, replay it, sync.
// B: capture a dispatch the runtime refuses (a null x), replay it, copy, sync.  This is the driver's shape.
// C: control, capture a real copy so the graph holds a node, replay it, sync.
#include <cuda_runtime.h>

#include "strata/kernels/elementwise.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata;

namespace {

int g_fail = 0;

void step(const char* what, cudaError_t e) {
    std::fflush(stdout);
    if (e != cudaSuccess) {
        std::printf("  FAIL %s: %s\n", what, cudaGetErrorString(e));
        ++g_fail;
    } else {
        std::printf("  ok   %s\n", what);
    }
    std::fflush(stdout);
}

void case_capture(const char* name, bool refused_kernel, bool real_copy) {
    std::printf("%s\n", name);
    float* d = nullptr;
    step("cudaMalloc 4096 B", cudaMalloc(&d, 4096));
    if (!d) return;
    cudaStream_t s = nullptr;
    step("cudaStreamCreateWithFlags", cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    float host[2] = {1.0f, 2.0f};
    if (real_copy) {
        step("cudaMemcpyAsync into the capture", cudaMemcpyAsync(d, host, sizeof host, cudaMemcpyHostToDevice, s));
    }

    cudaGraph_t g = nullptr;
    cudaGraphExec_t x = nullptr;
    step("cudaStreamBeginCapture", cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    if (refused_kernel) {
        // A null x is refused by the runtime before anything reaches the GPU, so the capture records nothing.
        kernels::scale_inplace(nullptr, 1024, 2.0f, s);
    }
    step("cudaStreamEndCapture", cudaStreamEndCapture(s, &g));
    step("cudaGraphInstantiate", cudaGraphInstantiate(&x, g, 0));
    step("cudaGraphLaunch", cudaGraphLaunch(x, s));
    step("cudaMemcpyAsync after the replay", cudaMemcpyAsync(host, d, sizeof host, cudaMemcpyDeviceToHost, s));
    step("cudaStreamSynchronize", cudaStreamSynchronize(s));
    step("cudaGraphExecDestroy", cudaGraphExecDestroy(x));
    step("cudaGraphDestroy", cudaGraphDestroy(g));
    step("cudaStreamDestroy", cudaStreamDestroy(s));
    cudaFree(d);
}

// D: the driver's shape.  Its embedding section runs 108 row cases, each one a refused dispatch (an interior
// pointer) plus two async copies and a sync, and only then does it capture.  Cases A to B show that an empty capture
// replays fine on a fresh stream, so what is reproduced here is the capture AFTER that much accumulated stream state:
// many chain slots cycled, many fences retired, a timeline that has been advanced 108 times.  Still no compute
// kernel, so nothing here can wedge a workgroup.
void case_accumulated_then_capture(int rounds) {
    std::printf("case D: %d refused dispatch plus copies and a sync, then a capture\n", rounds);
    float* d = nullptr;
    step("cudaMalloc 4096 B", cudaMalloc(&d, 4096));
    if (!d) return;
    cudaStream_t s = nullptr;
    step("cudaStreamCreateWithFlags", cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float host[2] = {1.0f, 2.0f};
    for (int i = 0; i < rounds; ++i) {
        kernels::scale_inplace(nullptr, 1024, 2.0f, s);                                              // refused
        if (cudaMemcpyAsync(d, host, sizeof host, cudaMemcpyHostToDevice, s) != cudaSuccess) {
            std::printf("  FAIL copy in round %d\n", i);
            ++g_fail;
            break;
        }
        if (cudaMemcpyAsync(host, d, sizeof host, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
            std::printf("  FAIL copy back in round %d\n", i);
            ++g_fail;
            break;
        }
        if (cudaStreamSynchronize(s) != cudaSuccess) {
            std::printf("  FAIL sync in round %d\n", i);
            ++g_fail;
            break;
        }
        if ((i & 15) == 15) std::printf("  ok   round %d\n", i);
    }
    cudaGraph_t g = nullptr;
    cudaGraphExec_t x = nullptr;
    step("cudaStreamBeginCapture after the rounds", cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    kernels::scale_inplace(nullptr, 1024, 2.0f, s);                                                 // refused inside
    step("cudaStreamEndCapture", cudaStreamEndCapture(s, &g));
    step("cudaGraphInstantiate", cudaGraphInstantiate(&x, g, 0));
    step("cudaGraphLaunch", cudaGraphLaunch(x, s));
    step("cudaMemcpyAsync after the replay", cudaMemcpyAsync(host, d, sizeof host, cudaMemcpyDeviceToHost, s));
    step("cudaStreamSynchronize", cudaStreamSynchronize(s));
    step("cudaGraphExecDestroy", cudaGraphExecDestroy(x));
    step("cudaGraphDestroy", cudaGraphDestroy(g));
    step("cudaStreamDestroy", cudaStreamDestroy(s));
    cudaFree(d);
}

// E: the refusal the driver actually takes.  Cases B and D refused a NULL pointer, which the host kernel body turns
// down before it ever reaches launch_kernel.  The driver's embedding_gather and scale_inplace(out + 1, ...) reach
// launch_kernel and are refused later, inside the binding step, because the offset into the allocation is not a
// multiple of minStorageBufferOffsetAlignment.  If the runtime has already opened a command buffer by then, the
// refusal leaves it begun and never ended, which is exactly the kind of state a later wait can hang on.  Run plain
// first (F), then inside a capture (E), which is the driver's shape.
void case_interior_refusal(const char* name, bool inside_capture) {
    std::printf("%s\n", name);
    float* d = nullptr;
    step("cudaMalloc 4096 B", cudaMalloc(&d, 4096));
    if (!d) return;
    cudaStream_t s = nullptr;
    step("cudaStreamCreateWithFlags", cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float host[2] = {1.0f, 2.0f};
    step("cudaMemcpyAsync to fill the buffer", cudaMemcpyAsync(d, host, sizeof host, cudaMemcpyHostToDevice, s));
    step("cudaStreamSynchronize", cudaStreamSynchronize(s));

    float* mid = reinterpret_cast<float*>(reinterpret_cast<char*>(d) + 4);          // 4 B in: not 64 aligned
    cudaGraph_t g = nullptr;
    if (inside_capture) step("cudaStreamBeginCapture", cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    kernels::scale_inplace(mid, 1024, 2.0f, s);                                      // refused while binding
    if (inside_capture) step("cudaStreamEndCapture", cudaStreamEndCapture(s, &g));
    step("cudaStreamSynchronize after the interior refusal", cudaStreamSynchronize(s));
    step("cudaMemcpyAsync reading the buffer back", cudaMemcpyAsync(host, d, sizeof host, cudaMemcpyDeviceToHost, s));
    step("cudaStreamSynchronize", cudaStreamSynchronize(s));
    if (inside_capture) {
        cudaGraphExec_t x = nullptr;
        step("cudaGraphInstantiate", cudaGraphInstantiate(&x, g, 0));
        step("cudaGraphLaunch", cudaGraphLaunch(x, s));
        step("cudaStreamSynchronize after the replay", cudaStreamSynchronize(s));
        step("cudaGraphExecDestroy", cudaGraphExecDestroy(x));
        step("cudaGraphDestroy", cudaGraphDestroy(g));
    }
    step("cudaStreamDestroy", cudaStreamDestroy(s));
    cudaFree(d);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);          // a killed process must still show its last step
    std::printf("vk_capture_probe\n");
    case_capture("case A: a capture that recorded nothing", false, false);
    case_capture("case B: a capture whose only kernel was refused", true, false);
    case_capture("case C: a capture holding a real copy node", false, true);
    if (const char* r = std::getenv("VK_CAPTURE_PROBE_ROUNDS")) case_accumulated_then_capture(std::atoi(r));
    if (const char* m = std::getenv("VK_CAPTURE_PROBE_MID")) {
        case_interior_refusal("case F: an interior pointer refused on a live stream", false);
        case_interior_refusal("case E: an interior pointer refused inside a capture", true);
        (void) m;
    }
    std::printf("vk_capture_probe: %d failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
