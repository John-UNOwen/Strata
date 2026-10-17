// vulkan/probe/stub_probe.cpp - verification probe for the generated stub path (S4).
//
// vk_caps_probe, in this same directory, asks the Vulkan driver what it can do.  This one asks the *link* what it
// answered: it calls, through its real declaration in include/strata/kernels/elementwise.hpp:43, an entry point that
// vulkan/tools/gen_stubs.py still has to fill - strata::kernels::scale_inplace.  Nothing else is called: no device, no
// model, no kernel.  The definition the call lands on is the generated stub, so this program must print that name on
// stderr and exit 3.  If it ever gets past the call, a real Vulkan scale_inplace exists, the generator no longer emits
// that stub, and this probe has to be retired with it.
//
// It used to call strata::kernels::add_inplace - the first symbol MSVC's linker reported unresolved, out of
// strata_vulkan_host.lib(expert_source.obj), and the entry point the probe was written against.  add_inplace now has a
// real Vulkan body (vulkan/src/kernels/elementwise.cpp over vulkan/shaders/elementwise_add.comp) and its stub is gone,
// so the probe moved to the next entry point in the same header that is still a stub.  Its contract did not change:
// the stub's name on stderr, exit 3.  The run that shows the OTHER half - that a ported entry point no longer reaches a
// stub and really dispatches - is strata_vk_parity (vulkan/probe/vk_add_parity.cpp).
#include <cstdio>

#include "strata/kernels/elementwise.hpp"

int main() {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    strata::kernels::scale_inplace(x, 4, 2.0f, nullptr);
    std::fprintf(stderr, "stub_probe: strata::kernels::scale_inplace RETURNED - no stub fired, probe is stale\n");
    return 1;
}
