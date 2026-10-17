// vulkan/src/engine/missing.cpp - the one thing every generated stub in vulkan/src/engine/stubs.cpp does.
//
// A stub is a stop with a label, not a result.  The engine reaches an entry point that has no Vulkan implementation
// yet, this prints exactly which one it reached, and the process ends with a non-zero exit code so nothing downstream
// - a serve loop, a launcher, a benchmark - can mistake "not ported" for an answer.
//
// The exit code is 3: 1 is what the engine's own argument errors return (src/program/generate.cpp's `std::exit(2)`
// for a missing value, `return 2` for a bad token list), and 2 is therefore already taken; 3 is the code that means
// "a stub was reached" and is quoted as such in VULKAN-BACKEND-PLAN.md's Phase 1 exit.
//
// Generated stubs name the symbol MSVC's linker could not resolve; see vulkan/tools/gen_stubs.py for how the list is
// produced from a failed link, and vulkan/src/engine/stubs.cpp's header for why the file must be regenerated, not edited.
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan_engine {

[[noreturn]] void missing(const char* name) {
    std::fflush(stdout);
    std::fprintf(stderr, "strata vulkan: stub reached: no Vulkan implementation for \"%s\" "
                         "(vulkan/src/engine/stubs.cpp)\n", name);
    std::fflush(stderr);
    std::exit(3);
}

}  // namespace strata::vulkan_engine
