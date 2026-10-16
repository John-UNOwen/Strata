// include/strata/core/win_dxgi.hpp - Windows port: what DXGI knows about the card a SYCL device runs on.  Level Zero
// exposes no LUID, so the device is found by its name among the DXGI adapters (the Intel driver reports both alike,
// e.g. "Intel(R) Arc(TM) A770 Graphics").  Two adapters of the same name cannot be told apart that way: no match.
#pragma once
#if defined(_WIN32)
#include <cstdint>
#include <string>

namespace strata::core {
/// The card's local (VRAM) segment for this process (IDXGIAdapter3::QueryVideoMemoryInfo): `budget`, what the OS lets
/// the process use before it starts paging, and `usage`, what the process holds there now.  False when no unique
/// adapter has the device's name or the query fails.
bool dxgi_local_memory(const std::string& device_name, uint64_t& budget, uint64_t& usage);
}  // namespace strata::core
#endif
