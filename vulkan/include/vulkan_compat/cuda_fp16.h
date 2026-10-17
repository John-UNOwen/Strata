// vulkan/include/vulkan_compat/cuda_fp16.h - CUDA's fp16 names, on Vulkan.
//
// Justified by iteration 0:
//   src\kernels\qsa_prompt_attn_parity.cpp(17,10): error C1083: Cannot open include file: 'cuda_fp16.h': No such file or directory
// and by what that file then asks for (src/kernels/qsa_prompt_attn_parity.cpp:40-41): a 16-bit type whose bits are the
// encoding, plus __half2float / __float2half to move between it and float.
//
// The 16-bit pattern is the same object on both sides: CUDA's __half is IEEE-754 binary16 in memory, and so is every
// fp16 value this port puts in a VkDeviceMemory block or reads back (SPIR-V's OpTypeFloat 16 is the same encoding), so
// the conversion here is a host-side bit question, not a device one.
//
// It maps to the tree's OWN converter, not to an invention of this header: include/strata/kernels/f16_bits.hpp holds
// the round-to-nearest-even f32 -> fp16 and the exact fp16 -> f32 that the CUDA kernels were moved onto after
// __float2half produced wrong bits (f16_bits.hpp:3-12 records the round-193 finding, and the converter is checked
// there against numpy's float16 vectors).  Same rule the kernels follow: bits first, no implicit rounding.
#pragma once

#include <cstdint>
#include <cstring>

#include "strata/kernels/f16_bits.hpp"   // strata::kernels::f16_from_f32 / f32_from_f16

namespace strata::compat_fp16 {

// Two bytes holding a binary16 value.  Deliberately not a `float`-like type with operators: the host code that names
// __half here stores and moves the encoding, and an implicit arithmetic type is how a rounding difference becomes
// invisible.  (On CUDA this header's type is the same two bytes, with the arithmetic coming from the device pass.)
struct __half_storage { std::uint16_t bits = 0; };

}  // namespace strata::compat_fp16

using __half = strata::compat_fp16::__half_storage;
using half = __half;

inline __half __ushort_as_half(std::uint16_t bits) { __half h; h.bits = bits; return h; }
inline std::uint16_t __half_as_ushort(__half h) { return h.bits; }
inline float __half2float(__half h) { return strata::kernels::f32_from_f16(h.bits); }
inline __half __float2half(float f) { return __ushort_as_half(strata::kernels::f16_from_f32(f)); }
inline __half __float2half_rn(float f) { return __float2half(f); }   // CUDA's explicit round-to-nearest-even spelling
