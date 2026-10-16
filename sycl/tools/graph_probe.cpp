// sycl/tools/graph_probe.cpp - does a SYCL graph run on this card?  (Windows two-card debugging, HANDOFF section 12.)
//
// The engine records its decode windows, commits and MTP steps as SYCL graphs.  On the Windows B580 + A770 PC every
// two-card failure followed graph work on the A770.  This probe runs the same pattern in miniature, on one context that
// holds every visible card (as dpct's default context does): three kernels eagerly, then the same three recorded as a
// graph and replayed, waiting after each replay, then replayed back to back.  Every step prints before it starts, so a
// hang names its step.
//
//   graph_probe [device index among the visible GPUs, default: the last] [replays, default 20]
//
// Build with the same icx line as the other probes (-fsycl).  Run it with the engine's environment
// (UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1), and A/B one variable per run: the selector (level_zero:1 alone, or
// level_zero:0;level_zero:1 with index 1) and SYCL_UR_USE_LEVEL_ZERO_V2=0.  Each kernel runs for microseconds: if a
// step does not finish, the graph path itself is stuck, and the TDR watchdog resets the card after ~10 s.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;

static void say(const char* fmt, double v = 0.0) {
    std::printf(fmt, v);
    std::printf("\n");
    std::fflush(stdout);
}

int main(int argc, char** argv) try {
    const std::vector<sycl::device> devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) { say("no GPU visible"); return 1; }
    for (size_t i = 0; i < devs.size(); ++i)
        std::printf("GPU %zu: %s (driver %s)\n", i, devs[i].get_info<sycl::info::device::name>().c_str(),
                    devs[i].get_info<sycl::info::device::driver_version>().c_str());
    const int idx = argc > 1 ? std::atoi(argv[1]) : (int) devs.size() - 1;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    if (idx < 0 || idx >= (int) devs.size() || reps < 1) { say("bad arguments"); return 2; }
    const sycl::device d = devs[(size_t) idx];
    std::printf("testing GPU %d, %d replays, SYCL_UR_USE_LEVEL_ZERO_V2=%s\n", idx, reps,
                std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2") ? std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2") : "(unset)");
    std::fflush(stdout);

    const sycl::context ctx{devs};   // one context for every visible card, as the engine has
    sycl::queue q{ctx, d, {sycl::property::queue::in_order{}}};
    constexpr size_t N = 1 << 20;
    int* buf = sycl::malloc_device<int>(N, q);
    q.memset(buf, 0, N * sizeof(int)).wait();
    auto three = [&](sycl::queue& qq) {
        for (int k = 0; k < 3; ++k) qq.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { buf[i] += 1; });
    };
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); };

    say("1. eager: 3 kernels ...");
    Clock::time_point t = Clock::now();
    three(q);
    q.wait();
    say("   done in %.2f ms", ms(t));

    say("2. graph: record + finalize ...");
    sx::command_graph g{ctx, d};
    g.begin_recording(q);
    three(q);
    g.end_recording();
    auto exec = g.finalize();
    say("   done");

    say("3. graph: replay, wait after each ...");
    for (int r = 0; r < reps; ++r) {
        t = Clock::now();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::printf("   replay %d: %.2f ms\n", r, ms(t));
        std::fflush(stdout);
    }

    say("4. graph: replays back to back, one wait ...");
    t = Clock::now();
    for (int r = 0; r < reps; ++r) q.ext_oneapi_graph(exec);
    q.wait();
    say("   done in %.2f ms", ms(t));

    say("5. eager after graph: 3 kernels ...");
    t = Clock::now();
    three(q);
    q.wait();
    say("   done in %.2f ms", ms(t));

    int first = 0, last = 0;
    q.memcpy(&first, buf, sizeof(int)).wait();
    q.memcpy(&last, buf + N - 1, sizeof(int)).wait();
    const int want = 3 + 3 * reps + 3 * reps + 3;
    std::printf("check: buf[0] = %d, buf[N-1] = %d, want %d: %s\n", first, last, want,
                first == want && last == want ? "PASS" : "FAIL");
    sycl::free(buf, q);
    return first == want && last == want ? 0 : 3;
} catch (const sycl::exception& e) {
    std::printf("sycl exception: %s\n", e.what());
    return 4;
}
