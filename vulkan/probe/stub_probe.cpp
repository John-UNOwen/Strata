// vulkan/probe/stub_probe.cpp - verification probe for the generated stub path (S4).
//
// vk_caps_probe, in this same directory, asks the Vulkan driver what it can do.  This one asks the *link* what it
// answered: it calls, through its real declaration in include/strata/kernels/elementwise.hpp:52, the first kernel
// entry point that vulkan/tools/gen_stubs.py had to fill - strata::kernels::add_inplace, reported unresolved by
// link.exe out of strata_vulkan_host.lib(expert_source.obj), referenced in strata::core::expert_hit_run.  Nothing
// else is called: no device, no model, no kernel.  The definition the call lands on is the generated stub, so this
// program must print that name on stderr and exit 3.  If it ever gets past the call, a real Vulkan add_inplace
// exists, the generator no longer emits that stub, and this probe has to be retired with it.
#include <cstdio>

#include "strata/kernels/elementwise.hpp"

int main() {
    float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    strata::kernels::add_inplace(dst, src, 4, nullptr);
    std::fprintf(stderr, "stub_probe: strata::kernels::add_inplace RETURNED - no stub fired, probe is stale\n");
    return 1;
}
