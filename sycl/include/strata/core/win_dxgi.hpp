// include/strata/core/win_dxgi.hpp - Windows port: what DXGI knows about the card a SYCL device runs on.  Level Zero
// exposes no LUID, so the device is found by its name among the DXGI adapters (the Intel driver reports both alike,
// e.g. "Intel(R) Arc(TM) A770 Graphics").  A name can match several adapters (the Intel driver registers an A770
// twice; two identical cards share a name): the answer is used only when all of them report the same figures.
#pragma once
#if defined(_WIN32)
#include <cstdint>
#include <string>

namespace strata::core {
/// The card's local (VRAM) segment for this process (IDXGIAdapter3::QueryVideoMemoryInfo): `budget`, what the OS lets
/// the process use before it starts paging, and `usage`, what the process holds there now.  False when no adapter has
/// the device's name, the adapters of that name report different figures, or the query fails.
bool dxgi_local_memory(const std::string& device_name, uint64_t& budget, uint64_t& usage);
}  // namespace strata::core
#endif
