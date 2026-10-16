// graph_probe.cpp - Level Zero command_graph capture/replay at a selector position.
// graph_probe <pos> [replays]: the same graph path the engine's window uses
// (command_graph<modifiable>, begin/end_recording on a dedicated queue, finalize,
// ext_oneapi_graph), ~2400 nodes per replay so a capture or replay failure shows
// up at the engine's scale. Every step prints before it starts, flushed.
#include <sycl/sycl.hpp>
#include "dpct/dpct.hpp"
#include "dpct/graph.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace sycl;

static void step(const std::string& msg) { std::printf("%s\n", msg.c_str()); std::fflush(stdout); }

int main(int argc, char** argv) {
    const int pos = argc > 1 ? std::atoi(argv[1]) : 0;
    const int replays = argc > 2 ? std::atoi(argv[2]) : 100;
    const int nodes = 2400; // the engine's window graph size

    step("graph_probe: selector position " + std::to_string(pos) +
         ", " + std::to_string(replays) + " replays");
    auto all = device::get_devices(); // filtered by ONEAPI_DEVICE_SELECTOR
    std::vector<device> gpus;
    for (auto& d : all)
        if (d.is_gpu()) gpus.push_back(d);
    if (pos < 0 || pos >= (int)gpus.size()) {
        std::printf("only %d GPUs visible\n", (int)gpus.size());
        return 2;
    }
    device dev = gpus[pos];
    step("device: " + dev.get_info<info::device::name>() +
         " | driver " + dev.get_info<info::device::driver_version>());

    step("step: create dedicated queue");
    // the engine's cs_ is dpct create_queue(true): an in-order queue
    queue q(dev, property::queue::in_order{});

    int* out = (int*)malloc_host(1, q); // mapped sentinel, like the engine's flags
    float* buf = malloc_device<float>(nodes + 1, q);

    step("step: warm kernel + buffers");
    q.parallel_for(nodes, [=](id<1> i) { buf[i] = 0.0f; }).wait();
    *out = 0;

    step("step: begin_recording " + std::to_string(nodes) + " nodes");
    // exactly the engine path: dpct::experimental::begin_recording on a dedicated
    // queue (command_graph<modifiable>(context, device)), submits, end_recording,
    // then command_graph<executable>(graph->finalize()) and ext_oneapi_graph
    dpct::experimental::begin_recording(&q);
    for (int n = 0; n < nodes; ++n)
        q.submit([=](handler& h) {
            h.single_task([=]() {
                buf[n] = buf[n] + 1.0f;
                if (n == nodes - 1) *out = *out + 1; // one bump per replay
            });
        });
    step("step: end_recording");
    dpct::experimental::command_graph_ptr graph = nullptr;
    dpct::experimental::end_recording(&q, &graph);
    if (graph == nullptr) { std::printf("end_recording produced no graph\n"); return 4; }

    step("step: finalize");
    dpct::experimental::command_graph_exec_ptr exec =
        new ext::oneapi::experimental::command_graph<
            ext::oneapi::experimental::graph_state::executable>(graph->finalize());
    delete graph;

    step("step: replay loop");
    for (int r = 0; r < replays; ++r) {
        step("replay " + std::to_string(r + 1) + "/" + std::to_string(replays));
        q.ext_oneapi_graph(*exec);
        q.wait();
        if (*out != r + 1) {
            std::printf("sentinel after replay %d: %d (expected %d)\n", r + 1, *out, r + 1);
            return 3;
        }
    }
    step("done: all replays finished with the sentinel correct");
    return 0;
}
