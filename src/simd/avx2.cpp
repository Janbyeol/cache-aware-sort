#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
#include "kernels.hpp"
#include <immintrin.h>
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to=function)
#elif defined(__GNUC__)
#pragma GCC target("avx2")
#endif
#include "network.hpp"

namespace hsort::simd {
namespace {
struct Avx2 {
    using Vec = __m256i;
    static Vec select(Vec mask, Vec yes, Vec no) noexcept { return _mm256_blendv_epi8(no, yes, mask); }
    static Vec min32(Vec a, Vec b) noexcept { return _mm256_min_epi32(a, b); }
    static Vec max32(Vec a, Vec b) noexcept { return _mm256_max_epi32(a, b); }
    static Vec greater_key(Vec a, Vec b) noexcept {
        return _mm256_shuffle_epi32(_mm256_cmpgt_epi32(a, b), _MM_SHUFFLE(2, 2, 0, 0));
    }
    template<int Xor>
    static Vec permute_xor(Vec v) noexcept {
        const Vec index = _mm256_setr_epi32(0 ^ Xor, 1 ^ Xor, 2 ^ Xor, 3 ^ Xor,
                                            4 ^ Xor, 5 ^ Xor, 6 ^ Xor, 7 ^ Xor);
        return _mm256_permutevar8x32_epi32(v, index);
    }
    template<unsigned Mask>
    static Vec blend(Vec low, Vec high) noexcept { return _mm256_blend_epi32(low, high, Mask); }
};
} // namespace
constinit const Kernels avx2{Network<Avx2>::full_leaf, Network<Avx2>::merge};
} // namespace hsort::simd
#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif
