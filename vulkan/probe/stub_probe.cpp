// vulkan/probe/stub_probe.cpp - verification probe for the generated stub path (S4).
//
// vk_caps_probe, in this same directory, asks the Vulkan driver what it can do.  This one asks the *link* what it
// answered: it calls, through its real declaration in include/strata/kernels/elementwise.hpp, an entry point that
// vulkan/tools/gen_stubs.py still has to fill - strata::kernels::scatter_rows_f32.  Nothing else is called: no device,
// no model, no kernel.  The definition the call lands on is the generated stub, so this program must print that name on
// stderr and exit 3.  If it ever gets past the call, a real Vulkan scatter_rows_f32 exists, the generator no longer
// emits that stub, and this probe has to be retired with it.
//
// It used to call strata::kernels::add_inplace, then strata::kernels::scale_inplace - each was the next entry point in
// this header that was still a stub.  Both now have real Vulkan bodies in vulkan/src/kernels/elementwise.cpp over their
// .comp in vulkan/shaders/, and their stubs are gone, which is why the probe moved again.  Its contract did not change:
// the stub's name on stderr, exit 3.  The run that shows the OTHER half - that a ported entry point no longer reaches a
// stub and really dispatches - is strata_vk_parity (vulkan/probe/vk_add_parity.cpp) and
// strata_parity_elementwise (../src/kernels/elementwise_parity.cpp, upstream's own driver).
#include <cstdio>

#include "strata/kernels/elementwise.hpp"

int main() {
    float src[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float dst[2] = {0.0f, 0.0f};
    const int idx[2] = {0, 3};
    strata::kernels::scatter_rows_f32(src, dst, idx, 2, 3, nullptr);
    std::fprintf(stderr, "stub_probe: strata::kernels::scatter_rows_f32 RETURNED - no stub fired, probe is stale\n");
    return 1;
}
