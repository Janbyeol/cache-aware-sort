#if defined(_M_ARM64) || (defined(__aarch64__) && defined(__ARM_NEON))
#include "kernels.hpp"
#if defined(_MSC_VER) && !defined(__clang__)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include "network.hpp"

namespace hsort::simd {
namespace {
struct Neon {
    using Vec = int32x4_t;
    static Vec select(uint32x4_t mask, Vec yes, Vec no) noexcept { return vbslq_s32(mask, yes, no); }
    static Vec min32(Vec a, Vec b) noexcept { return vminq_s32(a, b); }
    static Vec max32(Vec a, Vec b) noexcept { return vmaxq_s32(a, b); }
    static uint32x4_t greater_key(Vec a, Vec b) noexcept {
        const uint32x4_t cmp = vcgtq_s32(a, b);
        return vtrn1q_u32(cmp, cmp);
    }
    template<int Xor>
    static Vec permute_xor(Vec v) noexcept {
        if constexpr (Xor & 1) v = vrev64q_s32(v);
        if constexpr (Xor & 2) v = vextq_s32(v, v, 2);
        return v;
    }
    template<unsigned Mask>
    static Vec blend(Vec low, Vec high) noexcept {
        constexpr std::uint32_t words[] = {0u - (Mask & 1), 0u - ((Mask >> 1) & 1),
                                          0u - ((Mask >> 2) & 1), 0u - ((Mask >> 3) & 1)};
        return select(vld1q_u32(words), high, low);
    }
};
} // namespace
constinit const Kernels neon{Network<Neon>::full_leaf, Network<Neon>::merge};
} // namespace hsort::simd
#endif
