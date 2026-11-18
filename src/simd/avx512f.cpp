#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
#include "kernels.hpp"
#include <immintrin.h>
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f"))), apply_to=function)
#elif defined(__GNUC__)
#pragma GCC target("avx512f")
#endif
#include "network.hpp"

namespace hsort::simd {
namespace {
struct Avx512f {
    using Vec = __m512i;
    static Vec select(__mmask16 mask, Vec yes, Vec no) noexcept {
        return _mm512_mask_blend_epi32(mask, no, yes);
    }
    static Vec min32(Vec a, Vec b) noexcept { return _mm512_maskz_min_epi32(0xffff, a, b); }
    static Vec max32(Vec a, Vec b) noexcept { return _mm512_maskz_max_epi32(0xffff, a, b); }
    static __mmask16 greater_key(Vec a, Vec b) noexcept {
        const unsigned cmp = _mm512_cmpgt_epi32_mask(a, b) & 0x5555u;
        return static_cast<__mmask16>(cmp | (cmp << 1));
    }
    template<int Xor>
    static Vec permute_xor(Vec v) noexcept {
        const Vec index = _mm512_setr_epi32(0 ^ Xor, 1 ^ Xor, 2 ^ Xor, 3 ^ Xor,
                                            4 ^ Xor, 5 ^ Xor, 6 ^ Xor, 7 ^ Xor,
                                            8 ^ Xor, 9 ^ Xor, 10 ^ Xor, 11 ^ Xor,
                                            12 ^ Xor, 13 ^ Xor, 14 ^ Xor, 15 ^ Xor);
        return _mm512_maskz_permutexvar_epi32(0xffff, index, v);
    }
    template<unsigned Mask>
    static Vec blend(Vec low, Vec high) noexcept { return _mm512_mask_blend_epi32(Mask, low, high); }
};
} // namespace
constinit const Kernels avx512f{Network<Avx512f>::full_leaf, Network<Avx512f>::merge};
} // namespace hsort::simd
#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif
