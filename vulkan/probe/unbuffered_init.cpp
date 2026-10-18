// vulkan/probe/unbuffered_init.cpp - makes stdout unbuffered for a parity target, without touching the driver file.
//
// src/kernels/elementwise_parity.cpp prints its case results with std::printf.  Redirected to a file that is fully
// buffered, so a run killed by a timeout loses every line it printed and only the stderr messages survive, which
// makes the transcript useless for finding WHERE it stopped.  This translation unit runs a constructor before main
// and switches stdout to unbuffered.  The driver source stays byte-identical.
#include <cstdio>

namespace {

struct UnbufferedStdout {
    UnbufferedStdout() { std::setvbuf(stdout, nullptr, _IONBF, 0); }
};

UnbufferedStdout g_unbuffered_stdout;

}  // namespace
