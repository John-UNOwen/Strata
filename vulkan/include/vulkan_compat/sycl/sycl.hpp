// vulkan/include/vulkan_compat/sycl/sycl.hpp - the two SYCL type names one shared header asks for.
//
// Justified by iteration 0:
//   include\strata\core\native_head.hpp(10,10): error C1083: Cannot open include file: 'sycl/sycl.hpp': No such file or directory
//
// This one is not a CUDA name, and it is worth being exact about why it is here.  src/core/native_head.cpp is CUDA
// host code (it includes <cuda_runtime.h> at :7 and never names a SYCL type), but the header it compiles against,
// include/strata/core/native_head.hpp, carries SYCL member types the SYCL port put there: the include above at :10,
// `const void* table_for(sycl::queue& q) const;` at :66, and `std::optional<sycl::context> alloc_ctx_` plus
// `std::vector<std::pair<sycl::context, void*>> mirrors_` at :77-78 (commit 79e5b7f3, "mirror native embed table per
// queue context").  So the CUDA build - which compiles that same .cu at CMakeLists.txt:488 - hits the same wall:
// there is no sycl/sycl.hpp on a CUDA toolchain either.  The honest reading is that this is a cross-backend leak in
// the shared header, and the honest fix is in that header, which is outside this phase (vulkan/ only).
//
// What is provided here is therefore the minimum that makes the TU parse, and nothing more: `sycl::context` as a
// value type the two members can hold, and `sycl::queue` named only as a reference parameter.  This is NOT a SYCL
// runtime, no queue or context is ever created by this port, and `table_for` has exactly one caller in the tree -
// sycl/src/core/native_head.cpp - so the Vulkan build never reaches it.  Nothing here pretends the engine has USM
// contexts: Phase 0 measured what these cards do with imported host memory, and the Vulkan port uses that.
#pragma once

#include <cstddef>
#include <cstdint>

namespace sycl {

// Named only so `table_for(sycl::queue&)` (native_head.hpp:66) parses.  Never defined, never used here.
class queue;

// What native_head.hpp:77-78 hold by value: a copyable, comparable tag for "the allocation belongs to this device".
// On Vulkan the equivalent is which VkDevice an allocation was imported into (VULKAN-BACKEND-PLAN.md Phase 1's
// per-device table mirror); this shim carries the identity only.
class context {
public:
    context() = default;
    explicit context(int device) : device_(device) {}
    int device() const { return device_; }
    bool operator==(const context& o) const { return device_ == o.device_; }
    bool operator!=(const context& o) const { return device_ != o.device_; }

private:
    int device_ = -1;
};

}  // namespace sycl
