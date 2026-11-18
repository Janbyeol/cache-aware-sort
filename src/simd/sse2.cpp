#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
#include "kernels.hpp"
#include <emmintrin.h>
#include "network.hpp"

namespace hsort::simd {
namespace {
struct Sse2 {
    using Vec = __m128i;
    static Vec select(Vec mask, Vec yes, Vec no) noexcept {
        return _mm_or_si128(_mm_and_si128(mask, yes), _mm_andnot_si128(mask, no));
    }
    static Vec min32(Vec a, Vec b) noexcept { return select(_mm_cmpgt_epi32(a, b), b, a); }
    static Vec max32(Vec a, Vec b) noexcept { return select(_mm_cmpgt_epi32(a, b), a, b); }
    static Vec greater_key(Vec a, Vec b) noexcept {
        return _mm_shuffle_epi32(_mm_cmpgt_epi32(a, b), _MM_SHUFFLE(2, 2, 0, 0));
    }
    template<int Xor>
    static Vec permute_xor(Vec v) noexcept {
        return _mm_shuffle_epi32(v, _MM_SHUFFLE(3 ^ Xor, 2 ^ Xor, 1 ^ Xor, 0 ^ Xor));
    }
    template<unsigned Mask>
    static Vec blend(Vec low, Vec high) noexcept {
        return select(_mm_setr_epi32(-int(Mask & 1), -int((Mask >> 1) & 1),
                                     -int((Mask >> 2) & 1), -int((Mask >> 3) & 1)), high, low);
    }
};
} // namespace
constinit const Kernels sse2{Network<Sse2>::full_leaf, Network<Sse2>::merge};
} // namespace hsort::simd
#endif
