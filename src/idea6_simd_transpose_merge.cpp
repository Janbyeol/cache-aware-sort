#include "hybrid_sorts.hpp"

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
    #define HSORT_IDEA6_X64 1
    #include <immintrin.h>
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#elif defined(_M_ARM64) || (defined(__aarch64__) && defined(__ARM_NEON))
    #define HSORT_IDEA6_ARM64 1
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <arm64_neon.h>
    #else
        #include <arm_neon.h>
    #endif
#endif

namespace hsort {

namespace {

// ============================================================================
// [Idea 6] Transpose-Vectorized 256B Quad-Leaf + Dual-Chain Branchless SIMD Merge
//
// How Idea 6 evolved from Ideas 1-5:
//   - From Idea 1 & 4: Uses the 64B cache-line leaf (LEAF_SIZE) and 8 KiB L1
//     tile (TILE_SIZE) hierarchy with ping-pong buffer alternation.
//   - From Idea 3: Lifts the `O(1)` outer-boundary check (`src[begin] > src[mid]`)
//     and the bidirectional (front `->` and back `<-` simultaneous) branchless
//     merge principle directly into SIMD bitonic vector merging.
//   - Addressing Idea 5's bottlenecks:
//       1. Vertical Sorting Network + In-Register Matrix Transpose (256B Quad-Leaf):
//          Sorts 4 cache lines (4 * LEAF_SIZE = 256B) in registers using zero-shuffle
//          vertical column comparators followed by an 8x8 (or 4x4) register transpose
//          and odd-even vector merge. Eliminates the first two intra-tile merge passes
//          (width = LEAF_SIZE and 2 * LEAF_SIZE) while keeping the remaining 5 passes odd.
//       2. 1-Cycle In-Lane Shuffles & Single-Comparison Tie-Safe 8B Record Kernel:
//          Avoids 3-cycle cross-lane variable permutes (`vpermd`) and compares 8B
//          records with a single comparison per stage while preserving equal-key records.
//       3. Dual-Chain (Forward + Backward) Branchless SIMD Merge:
//          Breaks the serial loop-carried `carry` register dependency chain by running
//          `carry_fwd` and `carry_bwd` concurrently with `cmov` pointer updates.
//       4. Clean Cache-Line Preservation & Run-Boundary Short-Circuiting:
//          Skips memory stores on already-sorted 256B blocks to avoid dirtying cache
//          lines, and short-circuits merge passes when run boundaries are ordered.
//       5. Stack-Resident L1 Scratch for N <= TILE_SIZE (8 KiB) & ARM64 NEON SoA Leaf:
//          Uses `vld2q_s32`/`vst2q_s32` hardware key/ID de-interleaving on ARM64 NEON
//          for 4-lane 8B quad-leaf sorting paired with 64-bit `csel` bidirectional merge.
// ============================================================================

HSORT_ALWAYS_INLINE bool probe_and_handle_global_trend(Item* data, std::size_t n) noexcept {
    const std::size_t samples = std::min<std::size_t>(32, n - 1);
    std::size_t desc_adj   = 0;
    std::size_t desc_spans = 0;
    std::size_t prev_idx   = 0;

    for (std::size_t s = 0; s < samples; ++s) {
        const std::size_t idx = s * (n - 2) / (samples - 1);
        desc_adj += (data[idx] > data[idx + 1]);
        if (s > 0) desc_spans += (data[prev_idx] > data[idx]);
        prev_idx = idx;
    }

    const bool mostly_descending = (data[0] > data[n - 1]) &&
                                   (desc_adj * 8 >= samples * 7) &&
                                   (desc_spans * 8 >= (samples - 1) * 7);
    if (mostly_descending) {
        for (std::size_t l = 0, r = n - 1; l < r; ++l, --r) {
            if (data[l] > data[r]) std::swap(data[l], data[r]);
        }
    }

    const bool mostly_ascending = (data[0] <= data[n - 1]) &&
                                  ((samples - desc_adj) * 8 >= samples * 7) &&
                                  ((samples - 1 - desc_spans) * 8 >= (samples - 1) * 7);

    if (mostly_descending || mostly_ascending) {
        if (std::is_sorted(data, data + n)) return true;
    }
    return false;
}

HSORT_ALWAYS_INLINE bool all_run_boundaries_ordered(const Item* data,
                                                    std::size_t begin,
                                                    std::size_t end,
                                                    std::size_t width) noexcept {
    for (std::size_t boundary = begin + width; boundary < end; boundary += width) {
        if (data[boundary - 1] > data[boundary]) return false;
    }
    return true;
}

HSORT_ALWAYS_INLINE Item* branchless_merge_buffers(const Item* __restrict a, const Item* __restrict a_end,
                                                   const Item* __restrict b, const Item* __restrict b_end,
                                                   Item* __restrict out) noexcept {
    while (a < a_end && b < b_end) {
        const std::size_t safe = std::min<std::size_t>(
            static_cast<std::size_t>(a_end - a),
            static_cast<std::size_t>(b_end - b)
        );
        for (std::size_t s = 0; s < safe; ++s) {
            const Item va         = *a;
            const Item vb         = *b;
            const bool take_right = (vb < va);
            *out++                = take_right ? vb : va;
            a                    += !take_right;
            b                    += take_right;
        }
    }
    if (a < a_end) {
        const std::size_t rem = static_cast<std::size_t>(a_end - a);
        std::memcpy(out, a, rem * sizeof(Item));
        out += rem;
    } else if (b < b_end) {
        const std::size_t rem = static_cast<std::size_t>(b_end - b);
        std::memcpy(out, b, rem * sizeof(Item));
        out += rem;
    }
    return out;
}

HSORT_ALWAYS_INLINE void bidirectional_merge_disjoint(const Item* __restrict a, const Item* __restrict a_end,
                                                      const Item* __restrict b, const Item* __restrict b_end,
                                                      Item* __restrict out) noexcept {
    const std::size_t total = static_cast<std::size_t>(a_end - a) + static_cast<std::size_t>(b_end - b);
    const Item* l_fwd = a;
    const Item* l_bwd = a_end;
    const Item* r_fwd = b;
    const Item* r_bwd = b_end;
    Item* out_fwd     = out;
    Item* out_bwd     = out + total;

    while (l_fwd < l_bwd && r_fwd < r_bwd) {
        const std::size_t safe_pairs = std::min<std::size_t>(
            static_cast<std::size_t>(l_bwd - l_fwd),
            static_cast<std::size_t>(r_bwd - r_fwd)
        );
        for (std::size_t s = 0; s < safe_pairs; ++s) {
            const Item lv         = *l_fwd;
            const Item rv         = *r_fwd;
            const bool take_right = (rv < lv);
            *out_fwd++            = take_right ? rv : lv;
            l_fwd                += !take_right;
            r_fwd                += take_right;

            const Item lb        = *(l_bwd - 1);
            const Item rb        = *(r_bwd - 1);
            const bool take_left = (lb > rb);
            *--out_bwd           = take_left ? lb : rb;
            l_bwd               -= take_left;
            r_bwd               -= !take_left;
        }
    }
    if (l_fwd < l_bwd) {
        std::memcpy(out_fwd, l_fwd, static_cast<std::size_t>(l_bwd - l_fwd) * sizeof(Item));
    } else if (r_fwd < r_bwd) {
        std::memcpy(out_fwd, r_fwd, static_cast<std::size_t>(r_bwd - r_fwd) * sizeof(Item));
    }
}

#if defined(HSORT_IDEA6_X64)

bool detect_avx2_support() noexcept {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuid(regs, 1);
    constexpr int osxsave_avx = (1 << 27) | (1 << 28);
    if ((regs[2] & osxsave_avx) != osxsave_avx) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2") != 0;
#endif
}

bool has_avx2() noexcept {
    static const bool supported = detect_avx2_support();
    return supported;
}

#if defined(__clang__)
    #pragma clang attribute push(__attribute__((target("avx2"))), apply_to=function)
#elif defined(__GNUC__)
    #pragma GCC push_options
    #pragma GCC target("avx2")
#endif

HSORT_ALWAYS_INLINE __m256i load_vec(const Item* ptr) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
}

HSORT_ALWAYS_INLINE void store_vec(Item* ptr, __m256i v) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v);
}

#ifndef BENCH_RECORD
// ============================================================================
// 4-Byte Key (int32_t) x86-64 AVX2 Kernels (8 lanes per 256-bit register)
// ============================================================================

HSORT_ALWAYS_INLINE void cas_vec_4b(__m256i& a, __m256i& b) noexcept {
    const __m256i lo = _mm256_min_epi32(a, b);
    const __m256i hi = _mm256_max_epi32(a, b);
    a = lo;
    b = hi;
}

HSORT_ALWAYS_INLINE __m256i bitonic_finish_8x32(__m256i v) noexcept {
    // Distance 4: cross-lane 128-bit swap
    {
        const __m256i peer = _mm256_permute4x64_epi64(v, _MM_SHUFFLE(1, 0, 3, 2));
        const __m256i lo   = _mm256_min_epi32(v, peer);
        const __m256i hi   = _mm256_max_epi32(v, peer);
        v = _mm256_blend_epi32(lo, hi, 0xF0);
    }
    // Distance 2: in-lane 1-cycle shuffle
    {
        const __m256i peer = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
        const __m256i lo   = _mm256_min_epi32(v, peer);
        const __m256i hi   = _mm256_max_epi32(v, peer);
        v = _mm256_blend_epi32(lo, hi, 0xCC);
    }
    // Distance 1: in-lane 1-cycle shuffle
    {
        const __m256i peer = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        const __m256i lo   = _mm256_min_epi32(v, peer);
        const __m256i hi   = _mm256_max_epi32(v, peer);
        v = _mm256_blend_epi32(lo, hi, 0xAA);
    }
    return v;
}

HSORT_ALWAYS_INLINE void merge_vec_4b(__m256i& low, __m256i& high) noexcept {
    const __m256i rev = _mm256_shuffle_epi32(
        _mm256_permute4x64_epi64(high, _MM_SHUFFLE(0, 1, 2, 3)),
        _MM_SHUFFLE(2, 3, 0, 1)
    );
    const __m256i l0 = _mm256_min_epi32(low, rev);
    const __m256i h0 = _mm256_max_epi32(low, rev);
    low  = bitonic_finish_8x32(l0);
    high = bitonic_finish_8x32(h0);
}

HSORT_ALWAYS_INLINE __m256i sort_single_vec_4b(__m256i v) noexcept {
    {
        const __m256i peer = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        v = _mm256_blend_epi32(_mm256_min_epi32(v, peer), _mm256_max_epi32(v, peer), 0xAA);
    }
    {
        const __m256i peer = _mm256_shuffle_epi32(v, _MM_SHUFFLE(0, 1, 2, 3));
        v = _mm256_blend_epi32(_mm256_min_epi32(v, peer), _mm256_max_epi32(v, peer), 0xCC);
    }
    {
        const __m256i peer = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        v = _mm256_blend_epi32(_mm256_min_epi32(v, peer), _mm256_max_epi32(v, peer), 0xAA);
    }
    {
        const __m256i peer = _mm256_shuffle_epi32(
            _mm256_permute4x64_epi64(v, _MM_SHUFFLE(0, 1, 2, 3)),
            _MM_SHUFFLE(2, 3, 0, 1)
        );
        v = _mm256_blend_epi32(_mm256_min_epi32(v, peer), _mm256_max_epi32(v, peer), 0xF0);
    }
    {
        const __m256i peer = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
        v = _mm256_blend_epi32(_mm256_min_epi32(v, peer), _mm256_max_epi32(v, peer), 0xCC);
    }
    {
        const __m256i peer = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        v = _mm256_blend_epi32(_mm256_min_epi32(v, peer), _mm256_max_epi32(v, peer), 0xAA);
    }
    return v;
}

HSORT_ALWAYS_INLINE void sort_single_leaf_avx2(Item* data) noexcept {
    if (data[0] <= data[7] && data[7] <= data[15] && data[0] <= data[1]) {
        const __m256i c0 = _mm256_cmpgt_epi32(load_vec(data), load_vec(data + 1));
        const __m256i c1 = _mm256_cmpgt_epi32(load_vec(data + 7), load_vec(data + 8));
        if (_mm256_testz_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c0, c1))) {
            return;
        }
    }
    __m256i v0 = sort_single_vec_4b(load_vec(data));
    __m256i v1 = sort_single_vec_4b(load_vec(data + 8));
    merge_vec_4b(v0, v1);
    store_vec(data, v0);
    store_vec(data + 8, v1);
}

HSORT_ALWAYS_INLINE bool is_quad_leaf_sorted_avx2(const Item* data) noexcept {
    if (!(data[0] <= data[15] && data[15] <= data[31] &&
          data[31] <= data[47] && data[47] <= data[63] && data[0] <= data[1])) {
        return false;
    }
    const __m256i c0 = _mm256_cmpgt_epi32(load_vec(data + 0),  load_vec(data + 1));
    const __m256i c1 = _mm256_cmpgt_epi32(load_vec(data + 8),  load_vec(data + 9));
    const __m256i c2 = _mm256_cmpgt_epi32(load_vec(data + 16), load_vec(data + 17));
    const __m256i c3 = _mm256_cmpgt_epi32(load_vec(data + 24), load_vec(data + 25));
    const __m256i or0123 = _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3));
    if (!_mm256_testz_si256(or0123, or0123)) return false;

    const __m256i c4 = _mm256_cmpgt_epi32(load_vec(data + 32), load_vec(data + 33));
    const __m256i c5 = _mm256_cmpgt_epi32(load_vec(data + 40), load_vec(data + 41));
    const __m256i c6 = _mm256_cmpgt_epi32(load_vec(data + 48), load_vec(data + 49));
    const __m256i c7 = _mm256_cmpgt_epi32(load_vec(data + 55), load_vec(data + 56));
    const __m256i or4567 = _mm256_or_si256(_mm256_or_si256(c4, c5), _mm256_or_si256(c6, c7));
    return _mm256_testz_si256(or4567, or4567) != 0;
}

HSORT_ALWAYS_INLINE void sort_quad_leaf_avx2(Item* data) noexcept {
    if (is_quad_leaf_sorted_avx2(data)) return;

    __m256i r0 = load_vec(data + 0);
    __m256i r1 = load_vec(data + 8);
    __m256i r2 = load_vec(data + 16);
    __m256i r3 = load_vec(data + 24);
    __m256i r4 = load_vec(data + 32);
    __m256i r5 = load_vec(data + 40);
    __m256i r6 = load_vec(data + 48);
    __m256i r7 = load_vec(data + 56);

    // 1. Optimal 8-wire vertical sorting network across r0..r7 (19 comparators, 0 shuffles)
    cas_vec_4b(r0, r2); cas_vec_4b(r1, r3); cas_vec_4b(r4, r6); cas_vec_4b(r5, r7);
    cas_vec_4b(r0, r4); cas_vec_4b(r1, r5); cas_vec_4b(r2, r6); cas_vec_4b(r3, r7);
    cas_vec_4b(r0, r1); cas_vec_4b(r2, r3); cas_vec_4b(r4, r5); cas_vec_4b(r6, r7);
    cas_vec_4b(r2, r4); cas_vec_4b(r3, r5);
    cas_vec_4b(r1, r4); cas_vec_4b(r3, r6);
    cas_vec_4b(r1, r2); cas_vec_4b(r3, r4); cas_vec_4b(r5, r6);

    // 2. In-register 8x8 32-bit matrix transpose -> r0..r7 become 8 horizontally sorted vectors
    {
        const __m256i t0 = _mm256_unpacklo_epi32(r0, r1);
        const __m256i t1 = _mm256_unpackhi_epi32(r0, r1);
        const __m256i t2 = _mm256_unpacklo_epi32(r2, r3);
        const __m256i t3 = _mm256_unpackhi_epi32(r2, r3);
        const __m256i t4 = _mm256_unpacklo_epi32(r4, r5);
        const __m256i t5 = _mm256_unpackhi_epi32(r4, r5);
        const __m256i t6 = _mm256_unpacklo_epi32(r6, r7);
        const __m256i t7 = _mm256_unpackhi_epi32(r6, r7);

        const __m256i s0 = _mm256_unpacklo_epi64(t0, t2);
        const __m256i s1 = _mm256_unpackhi_epi64(t0, t2);
        const __m256i s2 = _mm256_unpacklo_epi64(t1, t3);
        const __m256i s3 = _mm256_unpackhi_epi64(t1, t3);
        const __m256i s4 = _mm256_unpacklo_epi64(t4, t6);
        const __m256i s5 = _mm256_unpackhi_epi64(t4, t6);
        const __m256i s6 = _mm256_unpacklo_epi64(t5, t7);
        const __m256i s7 = _mm256_unpackhi_epi64(t5, t7);

        r0 = _mm256_permute2x128_si256(s0, s4, 0x20);
        r1 = _mm256_permute2x128_si256(s1, s5, 0x20);
        r2 = _mm256_permute2x128_si256(s2, s6, 0x20);
        r3 = _mm256_permute2x128_si256(s3, s7, 0x20);
        r4 = _mm256_permute2x128_si256(s0, s4, 0x31);
        r5 = _mm256_permute2x128_si256(s1, s5, 0x31);
        r6 = _mm256_permute2x128_si256(s2, s6, 0x31);
        r7 = _mm256_permute2x128_si256(s3, s7, 0x31);
    }

    // 3. Merge pairs into four 16-element sorted leaves
    merge_vec_4b(r0, r1);
    merge_vec_4b(r2, r3);
    merge_vec_4b(r4, r5);
    merge_vec_4b(r6, r7);

    // 4. Merge into two 32-element sorted runs: (r0, r2, r1, r3) and (r4, r6, r5, r7)
    merge_vec_4b(r0, r2); merge_vec_4b(r4, r6);
    merge_vec_4b(r1, r3); merge_vec_4b(r5, r7);
    merge_vec_4b(r2, r1); merge_vec_4b(r6, r5);

    __m256i a0 = r0, a1 = r2, a2 = r1, a3 = r3;
    __m256i b0 = r4, b1 = r6, b2 = r5, b3 = r7;

    // 5. Odd-Even 8-vector merge into one 64-element sorted quad-leaf
    merge_vec_4b(a0, b0); merge_vec_4b(a1, b1); merge_vec_4b(a2, b2); merge_vec_4b(a3, b3);
    merge_vec_4b(b0, a2); merge_vec_4b(b1, a3);
    merge_vec_4b(b0, a1); merge_vec_4b(a2, b1); merge_vec_4b(a3, b2);

    store_vec(data + 0,  a0);
    store_vec(data + 8,  b0);
    store_vec(data + 16, a1);
    store_vec(data + 24, a2);
    store_vec(data + 32, b1);
    store_vec(data + 40, a3);
    store_vec(data + 48, b2);
    store_vec(data + 56, b3);
}

inline void merge_simd_avx2(const Item* __restrict src, Item* __restrict dst,
                            std::size_t begin, std::size_t mid, std::size_t end) noexcept {
    constexpr std::ptrdiff_t LANES = 8;
    const std::size_t left_len  = mid - begin;
    const std::size_t right_len = end - mid;

    if (left_len < 2 * LANES || right_len < 2 * LANES) {
        detail::bidirectional_branchless_merge(src, dst, begin, mid, end);
        return;
    }

    const Item* l_fwd = src + begin;
    const Item* r_fwd = src + mid;
    const Item* l_bwd = src + mid;
    const Item* r_bwd = src + end;
    Item* out_fwd     = dst + begin;
    Item* out_bwd     = dst + end;

    __m256i low_init  = load_vec(l_fwd); l_fwd += LANES;
    __m256i carry_fwd = load_vec(r_fwd); r_fwd += LANES;
    merge_vec_4b(low_init, carry_fwd);
    store_vec(out_fwd, low_init);
    out_fwd += LANES;

    l_bwd -= LANES; __m256i carry_bwd = load_vec(l_bwd);
    r_bwd -= LANES; __m256i high_init = load_vec(r_bwd);
    merge_vec_4b(carry_bwd, high_init);
    out_bwd -= LANES;
    store_vec(out_bwd, high_init);

    while ((l_bwd - l_fwd) >= 2 * LANES && (r_bwd - r_fwd) >= 2 * LANES) {
        const bool take_lf        = !(*r_fwd < *l_fwd);
        const Item* const ptr_fwd = take_lf ? l_fwd : r_fwd;
        l_fwd                    += take_lf ? LANES : 0;
        r_fwd                    += take_lf ? 0 : LANES;
        __m256i v_fwd             = load_vec(ptr_fwd);

        const bool take_lb        = (*(l_bwd - 1) > *(r_bwd - 1));
        l_bwd                    -= take_lb ? LANES : 0;
        r_bwd                    -= take_lb ? 0 : LANES;
        const Item* const ptr_bwd = take_lb ? l_bwd : r_bwd;
        __m256i v_bwd             = load_vec(ptr_bwd);

        merge_vec_4b(v_fwd, carry_fwd);
        merge_vec_4b(carry_bwd, v_bwd);

        store_vec(out_fwd, v_fwd);
        out_fwd += LANES;
        out_bwd -= LANES;
        store_vec(out_bwd, v_bwd);
    }

    while ((l_bwd - l_fwd) >= LANES && (r_bwd - r_fwd) >= LANES) {
        const bool take_lf        = !(*r_fwd < *l_fwd);
        const Item* const ptr_fwd = take_lf ? l_fwd : r_fwd;
        l_fwd                    += take_lf ? LANES : 0;
        r_fwd                    += take_lf ? 0 : LANES;
        __m256i v_fwd             = load_vec(ptr_fwd);
        merge_vec_4b(v_fwd, carry_fwd);
        store_vec(out_fwd, v_fwd);
        out_fwd += LANES;
    }

    merge_vec_4b(carry_fwd, carry_bwd);
    alignas(32) Item carries[2 * LANES];
    store_vec(carries, carry_fwd);
    store_vec(carries + LANES, carry_bwd);

    if ((l_bwd - l_fwd) > (r_bwd - r_fwd)) {
        std::swap(l_fwd, r_fwd);
        std::swap(l_bwd, r_bwd);
    }

    alignas(32) Item pending[3 * LANES];
    Item* const pending_end = branchless_merge_buffers(
        carries, carries + 2 * LANES, l_fwd, l_bwd, pending
    );
    bidirectional_merge_disjoint(pending, pending_end, r_fwd, r_bwd, out_fwd);
}

#else
// ============================================================================
// 8-Byte Key+ID Record (DistanceRecord) x86-64 AVX2 Kernels (4 lanes per register)
// ============================================================================

HSORT_ALWAYS_INLINE void cas_vec_8b(__m256i& a, __m256i& b) noexcept {
    const __m256i cmp  = _mm256_cmpgt_epi32(a, b);
    const __m256i swap = _mm256_shuffle_epi32(cmp, _MM_SHUFFLE(2, 2, 0, 0));
    const __m256i lo   = _mm256_blendv_epi8(a, b, swap);
    const __m256i hi   = _mm256_blendv_epi8(b, a, swap);
    a = lo;
    b = hi;
}

template <int Mask>
HSORT_ALWAYS_INLINE __m256i bitonic_step_8b(__m256i v, __m256i peer) noexcept {
    const __m256i l    = _mm256_blend_epi32(v, peer, Mask);
    const __m256i r    = _mm256_blend_epi32(peer, v, Mask);
    const __m256i cmp  = _mm256_cmpgt_epi32(l, r);
    const __m256i swap = _mm256_shuffle_epi32(cmp, _MM_SHUFFLE(2, 2, 0, 0));
    return _mm256_blendv_epi8(v, peer, swap);
}

HSORT_ALWAYS_INLINE __m256i bitonic_finish_4x64(__m256i v) noexcept {
    v = bitonic_step_8b<0xF0>(v, _mm256_permute4x64_epi64(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = bitonic_step_8b<0xCC>(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    return v;
}

HSORT_ALWAYS_INLINE void merge_vec_8b(__m256i& low, __m256i& high) noexcept {
    __m256i rev = _mm256_permute4x64_epi64(high, _MM_SHUFFLE(0, 1, 2, 3));
    cas_vec_8b(low, rev);
    low  = bitonic_finish_4x64(low);
    high = bitonic_finish_4x64(rev);
}

HSORT_ALWAYS_INLINE __m256i sort_single_vec_8b(__m256i v) noexcept {
    v = bitonic_step_8b<0xCC>(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = bitonic_step_8b<0xF0>(v, _mm256_permute4x64_epi64(v, _MM_SHUFFLE(0, 1, 2, 3)));
    v = bitonic_step_8b<0xCC>(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    return v;
}

HSORT_ALWAYS_INLINE void sort_single_leaf_avx2(Item* data) noexcept {
    if (data[0] <= data[3] && data[3] <= data[7] && data[0] <= data[1]) {
        const __m256i key_mask = _mm256_set1_epi64x(0x00000000FFFFFFFFULL);
        const __m256i c0 = _mm256_cmpgt_epi32(load_vec(data), load_vec(data + 1));
        const __m256i c1 = _mm256_cmpgt_epi32(load_vec(data + 3), load_vec(data + 4));
        if (_mm256_testz_si256(_mm256_or_si256(c0, c1), key_mask)) {
            return;
        }
    }
    __m256i v0 = sort_single_vec_8b(load_vec(data));
    __m256i v1 = sort_single_vec_8b(load_vec(data + 4));
    merge_vec_8b(v0, v1);
    store_vec(data, v0);
    store_vec(data + 4, v1);
}

HSORT_ALWAYS_INLINE bool is_quad_leaf_sorted_avx2(const Item* data) noexcept {
    if (!(data[0] <= data[7] && data[7] <= data[15] &&
          data[15] <= data[23] && data[23] <= data[31] && data[0] <= data[1])) {
        return false;
    }
    const __m256i key_mask = _mm256_set1_epi64x(0x00000000FFFFFFFFULL);
    const __m256i c0 = _mm256_cmpgt_epi32(load_vec(data + 0),  load_vec(data + 1));
    const __m256i c1 = _mm256_cmpgt_epi32(load_vec(data + 4),  load_vec(data + 5));
    const __m256i c2 = _mm256_cmpgt_epi32(load_vec(data + 8),  load_vec(data + 9));
    const __m256i c3 = _mm256_cmpgt_epi32(load_vec(data + 12), load_vec(data + 13));
    const __m256i or0123 = _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3));
    if (!_mm256_testz_si256(or0123, key_mask)) return false;

    const __m256i c4 = _mm256_cmpgt_epi32(load_vec(data + 16), load_vec(data + 17));
    const __m256i c5 = _mm256_cmpgt_epi32(load_vec(data + 20), load_vec(data + 21));
    const __m256i c6 = _mm256_cmpgt_epi32(load_vec(data + 24), load_vec(data + 25));
    const __m256i c7 = _mm256_cmpgt_epi32(load_vec(data + 27), load_vec(data + 28));
    const __m256i or4567 = _mm256_or_si256(_mm256_or_si256(c4, c5), _mm256_or_si256(c6, c7));
    return _mm256_testz_si256(or4567, key_mask) != 0;
}

HSORT_ALWAYS_INLINE void sort_quad_leaf_avx2(Item* data) noexcept {
    if (is_quad_leaf_sorted_avx2(data)) return;

    __m256i r0 = load_vec(data + 0);
    __m256i r1 = load_vec(data + 4);
    __m256i r2 = load_vec(data + 8);
    __m256i r3 = load_vec(data + 12);
    __m256i r4 = load_vec(data + 16);
    __m256i r5 = load_vec(data + 20);
    __m256i r6 = load_vec(data + 24);
    __m256i r7 = load_vec(data + 28);

    // 1. Two parallel 4-wire vertical sorting networks (5 comparators each, 0 shuffles)
    cas_vec_8b(r0, r1); cas_vec_8b(r2, r3); cas_vec_8b(r4, r5); cas_vec_8b(r6, r7);
    cas_vec_8b(r0, r2); cas_vec_8b(r1, r3); cas_vec_8b(r4, r6); cas_vec_8b(r5, r7);
    cas_vec_8b(r1, r2);                     cas_vec_8b(r5, r6);

    // 2. Two parallel 4x4 64-bit matrix transposes -> r0..r7 become 8 horizontally sorted 4-record vectors
    {
        const __m256i t0 = _mm256_unpacklo_epi64(r0, r1);
        const __m256i t1 = _mm256_unpackhi_epi64(r0, r1);
        const __m256i t2 = _mm256_unpacklo_epi64(r2, r3);
        const __m256i t3 = _mm256_unpackhi_epi64(r2, r3);
        r0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        r1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        r2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        r3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        const __m256i u0 = _mm256_unpacklo_epi64(r4, r5);
        const __m256i u1 = _mm256_unpackhi_epi64(r4, r5);
        const __m256i u2 = _mm256_unpacklo_epi64(r6, r7);
        const __m256i u3 = _mm256_unpackhi_epi64(r6, r7);
        r4 = _mm256_permute2x128_si256(u0, u2, 0x20);
        r5 = _mm256_permute2x128_si256(u1, u3, 0x20);
        r6 = _mm256_permute2x128_si256(u0, u2, 0x31);
        r7 = _mm256_permute2x128_si256(u1, u3, 0x31);
    }

    // 3. Merge pairs into four 8-record sorted leaves
    merge_vec_8b(r0, r1);
    merge_vec_8b(r2, r3);
    merge_vec_8b(r4, r5);
    merge_vec_8b(r6, r7);

    // 4. Merge into two 16-record sorted runs: (r0, r2, r1, r3) and (r4, r6, r5, r7)
    merge_vec_8b(r0, r2); merge_vec_8b(r4, r6);
    merge_vec_8b(r1, r3); merge_vec_8b(r5, r7);
    merge_vec_8b(r2, r1); merge_vec_8b(r6, r5);

    __m256i a0 = r0, a1 = r2, a2 = r1, a3 = r3;
    __m256i b0 = r4, b1 = r6, b2 = r5, b3 = r7;

    // 5. Odd-Even 8-vector merge into one 32-record sorted quad-leaf
    merge_vec_8b(a0, b0); merge_vec_8b(a1, b1); merge_vec_8b(a2, b2); merge_vec_8b(a3, b3);
    merge_vec_8b(b0, a2); merge_vec_8b(b1, a3);
    merge_vec_8b(b0, a1); merge_vec_8b(a2, b1); merge_vec_8b(a3, b2);

    store_vec(data + 0,  a0);
    store_vec(data + 4,  b0);
    store_vec(data + 8,  a1);
    store_vec(data + 12, a2);
    store_vec(data + 16, b1);
    store_vec(data + 20, a3);
    store_vec(data + 24, b2);
    store_vec(data + 28, b3);
}

inline void merge_simd_avx2(const Item* __restrict src, Item* __restrict dst,
                            std::size_t begin, std::size_t mid, std::size_t end) noexcept {
    constexpr std::ptrdiff_t LANES = 4;
    const std::size_t left_len  = mid - begin;
    const std::size_t right_len = end - mid;

    if (left_len < 2 * LANES || right_len < 2 * LANES) {
        detail::bidirectional_branchless_merge(src, dst, begin, mid, end);
        return;
    }

    const Item* l_fwd = src + begin;
    const Item* r_fwd = src + mid;
    const Item* l_bwd = src + mid;
    const Item* r_bwd = src + end;
    Item* out_fwd     = dst + begin;
    Item* out_bwd     = dst + end;

    __m256i low_init  = load_vec(l_fwd); l_fwd += LANES;
    __m256i carry_fwd = load_vec(r_fwd); r_fwd += LANES;
    merge_vec_8b(low_init, carry_fwd);
    store_vec(out_fwd, low_init);
    out_fwd += LANES;

    l_bwd -= LANES; __m256i carry_bwd = load_vec(l_bwd);
    r_bwd -= LANES; __m256i high_init = load_vec(r_bwd);
    merge_vec_8b(carry_bwd, high_init);
    out_bwd -= LANES;
    store_vec(out_bwd, high_init);

    while ((l_bwd - l_fwd) >= 2 * LANES && (r_bwd - r_fwd) >= 2 * LANES) {
        const bool take_lf        = !(*r_fwd < *l_fwd);
        const Item* const ptr_fwd = take_lf ? l_fwd : r_fwd;
        l_fwd                    += take_lf ? LANES : 0;
        r_fwd                    += take_lf ? 0 : LANES;
        __m256i v_fwd             = load_vec(ptr_fwd);

        const bool take_lb        = (*(l_bwd - 1) > *(r_bwd - 1));
        l_bwd                    -= take_lb ? LANES : 0;
        r_bwd                    -= take_lb ? 0 : LANES;
        const Item* const ptr_bwd = take_lb ? l_bwd : r_bwd;
        __m256i v_bwd             = load_vec(ptr_bwd);

        merge_vec_8b(v_fwd, carry_fwd);
        merge_vec_8b(carry_bwd, v_bwd);

        store_vec(out_fwd, v_fwd);
        out_fwd += LANES;
        out_bwd -= LANES;
        store_vec(out_bwd, v_bwd);
    }

    while ((l_bwd - l_fwd) >= LANES && (r_bwd - r_fwd) >= LANES) {
        const bool take_lf        = !(*r_fwd < *l_fwd);
        const Item* const ptr_fwd = take_lf ? l_fwd : r_fwd;
        l_fwd                    += take_lf ? LANES : 0;
        r_fwd                    += take_lf ? 0 : LANES;
        __m256i v_fwd             = load_vec(ptr_fwd);
        merge_vec_8b(v_fwd, carry_fwd);
        store_vec(out_fwd, v_fwd);
        out_fwd += LANES;
    }

    merge_vec_8b(carry_fwd, carry_bwd);
    alignas(32) Item carries[2 * LANES];
    store_vec(carries, carry_fwd);
    store_vec(carries + LANES, carry_bwd);

    if ((l_bwd - l_fwd) > (r_bwd - r_fwd)) {
        std::swap(l_fwd, r_fwd);
        std::swap(l_bwd, r_bwd);
    }

    alignas(32) Item pending[3 * LANES];
    Item* const pending_end = branchless_merge_buffers(
        carries, carries + 2 * LANES, l_fwd, l_bwd, pending
    );
    bidirectional_merge_disjoint(pending, pending_end, r_fwd, r_bwd, out_fwd);
}

#endif // BENCH_RECORD

inline void sort_partial_quad_leaf_avx2(Item* data, Item* scratch, std::size_t rem) noexcept {
    if (rem <= LEAF_SIZE) {
        if (rem == LEAF_SIZE) sort_single_leaf_avx2(data);
        else                  detail::insertion_sort_leaf(data, rem);
        return;
    }

    for (std::size_t off = 0; off < rem; off += LEAF_SIZE) {
        const std::size_t len = std::min<std::size_t>(LEAF_SIZE, rem - off);
        if (len == LEAF_SIZE) sort_single_leaf_avx2(data + off);
        else                  detail::insertion_sort_leaf(data + off, len);
    }

    detail::bidirectional_branchless_merge(data, scratch, 0, LEAF_SIZE, std::min<std::size_t>(rem, 2 * LEAF_SIZE));
    if (rem > 2 * LEAF_SIZE) {
        const std::size_t mid2 = std::min<std::size_t>(rem, 3 * LEAF_SIZE);
        detail::bidirectional_branchless_merge(data, scratch, 2 * LEAF_SIZE, mid2, rem);
        detail::bidirectional_branchless_merge(scratch, data, 0, 2 * LEAF_SIZE, rem);
    } else {
        std::memcpy(data, scratch, rem * sizeof(Item));
    }
}

HSORT_ALWAYS_INLINE void pingpong_merge_pair_avx2(const Item* __restrict src, Item* __restrict dst,
                                                  std::size_t begin, std::size_t mid, std::size_t end) noexcept {
    if (mid >= end || src[mid - 1] <= src[mid]) {
        std::memcpy(dst + begin, src + begin, (end - begin) * sizeof(Item));
        return;
    }

    const std::size_t trim_left = (src[begin] > src[mid])
        ? begin
        : static_cast<std::size_t>(std::upper_bound(src + begin, src + mid, src[mid]) - src);

    const std::size_t trim_right = (src[end - 1] < src[mid - 1])
        ? end
        : static_cast<std::size_t>(std::lower_bound(src + mid, src + end, src[mid - 1]) - src);

    if (trim_left > begin) {
        std::memcpy(dst + begin, src + begin, (trim_left - begin) * sizeof(Item));
    }
    if (trim_right < end) {
        std::memcpy(dst + trim_right, src + trim_right, (end - trim_right) * sizeof(Item));
    }

    merge_simd_avx2(src, dst, trim_left, mid, trim_right);
}

inline void pingpong_merge_pass_avx2(const Item* __restrict src, Item* __restrict dst,
                                     std::size_t begin, std::size_t end, std::size_t width) noexcept {
    for (std::size_t first = begin; first < end; first += 2 * width) {
        const std::size_t mid  = std::min(end, first + width);
        const std::size_t last = std::min(end, mid + width);
        pingpong_merge_pair_avx2(src, dst, first, mid, last);
    }
}

void sort_with_scratch_avx2(Item* data, Item* scratch, std::size_t n) noexcept {
    constexpr std::size_t QUAD_LEAF = 4 * LEAF_SIZE;

    for (std::size_t tile_begin = 0; tile_begin < n; tile_begin += TILE_SIZE) {
        const std::size_t tile_end = std::min(n, tile_begin + TILE_SIZE);

        std::size_t first = tile_begin;
        for (; first + QUAD_LEAF <= tile_end; first += QUAD_LEAF) {
            sort_quad_leaf_avx2(data + first);
        }
        if (first < tile_end) {
            sort_partial_quad_leaf_avx2(data + first, scratch + first, tile_end - first);
        }

        Item* src = data;
        Item* dst = scratch;
        for (std::size_t width = QUAD_LEAF; width < tile_end - tile_begin; width *= 2) {
            if (all_run_boundaries_ordered(src, tile_begin, tile_end, width)) {
                break;
            }
            pingpong_merge_pass_avx2(src, dst, tile_begin, tile_end, width);
            std::swap(src, dst);
        }

        if (n <= TILE_SIZE) {
            if (src != data) {
                std::memcpy(data, src, n * sizeof(Item));
            }
            return;
        }

        if (src != scratch) {
            std::memcpy(scratch + tile_begin, src + tile_begin,
                        (tile_end - tile_begin) * sizeof(Item));
        }
    }

    Item* src = scratch;
    Item* dst = data;
    for (std::size_t width = TILE_SIZE; width < n; width *= 2) {
        if (all_run_boundaries_ordered(src, 0, n, width)) {
            break;
        }
        pingpong_merge_pass_avx2(src, dst, 0, n, width);
        std::swap(src, dst);
    }

    if (src != data) {
        std::memcpy(data, src, n * sizeof(Item));
    }
}

void dispatch_avx2_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        if (n == LEAF_SIZE) sort_single_leaf_avx2(data);
        else                detail::insertion_sort_leaf(data, n);
        return;
    }
    if (n == 4 * LEAF_SIZE) {
        sort_quad_leaf_avx2(data);
        return;
    }
    if (probe_and_handle_global_trend(data, n)) return;

    if (n <= TILE_SIZE) {
        alignas(CACHE_LINE_BYTES) Item stack_scratch[TILE_SIZE];
        sort_with_scratch_avx2(data, stack_scratch, n);
        return;
    }

    AlignedBuffer heap_scratch(n);
    sort_with_scratch_avx2(data, heap_scratch.data(), n);
}

#if defined(__clang__)
    #pragma clang attribute pop
#elif defined(__GNUC__)
    #pragma GCC pop_options
#endif

#elif defined(HSORT_IDEA6_ARM64)
// ============================================================================
// ARM64 NEON Architecture-Matched Kernels:
//   - 4B: 4x4 Vertical Sorting Network + 4x4 Transpose + Dual-Chain 4-Lane Merge
//   - 8B: Hardware `vld2q_s32`/`vst2q_s32` Key/ID De-Interleaving (4 records per
//         128-bit register pair instead of 2!) + 4x4 Transpose for Quad-Leaves,
//         paired with ARM64 64-bit `csel` `bidirectional_branchless_merge` to
//         avoid the 2-lane vector merge bottleneck.
// ============================================================================

HSORT_ALWAYS_INLINE void transpose_4x4_s32(int32x4_t& r0, int32x4_t& r1,
                                           int32x4_t& r2, int32x4_t& r3) noexcept {
    const int32x4_t t0 = vtrn1q_s32(r0, r1);
    const int32x4_t t1 = vtrn2q_s32(r0, r1);
    const int32x4_t t2 = vtrn1q_s32(r2, r3);
    const int32x4_t t3 = vtrn2q_s32(r2, r3);
    r0 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
    r1 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
    r2 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
    r3 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
}

#ifndef BENCH_RECORD

HSORT_ALWAYS_INLINE void cas_neon_4b(int32x4_t& a, int32x4_t& b) noexcept {
    const int32x4_t lo = vminq_s32(a, b);
    const int32x4_t hi = vmaxq_s32(a, b);
    a = lo;
    b = hi;
}

HSORT_ALWAYS_INLINE void merge_vec_neon_4b(int32x4_t& low, int32x4_t& high) noexcept {
    int32x4_t rev = vrev64q_s32(high);
    rev = vextq_s32(rev, rev, 2);
    cas_neon_4b(low, rev);

    int32x4_t x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(low), vreinterpretq_s64_s32(rev)));
    int32x4_t y = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(low), vreinterpretq_s64_s32(rev)));
    cas_neon_4b(x, y);

    int32x4_t even = vtrn1q_s32(x, y);
    int32x4_t odd  = vtrn2q_s32(x, y);
    cas_neon_4b(even, odd);

    const int32x4_t lo_pairs = vtrn1q_s32(even, odd);
    const int32x4_t hi_pairs = vtrn2q_s32(even, odd);
    low  = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(lo_pairs), vreinterpretq_s64_s32(hi_pairs)));
    high = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(lo_pairs), vreinterpretq_s64_s32(hi_pairs)));
}

HSORT_ALWAYS_INLINE void sort_single_leaf_neon(Item* data) noexcept {
    int32x4_t r0 = vld1q_s32(data + 0);
    int32x4_t r1 = vld1q_s32(data + 4);
    int32x4_t r2 = vld1q_s32(data + 8);
    int32x4_t r3 = vld1q_s32(data + 12);

    cas_neon_4b(r0, r1); cas_neon_4b(r2, r3);
    cas_neon_4b(r0, r2); cas_neon_4b(r1, r3);
    cas_neon_4b(r1, r2);
    transpose_4x4_s32(r0, r1, r2, r3);

    merge_vec_neon_4b(r0, r1);
    merge_vec_neon_4b(r2, r3);
    merge_vec_neon_4b(r0, r2);
    merge_vec_neon_4b(r1, r3);
    merge_vec_neon_4b(r2, r1);

    vst1q_s32(data + 0,  r0);
    vst1q_s32(data + 4,  r2);
    vst1q_s32(data + 8,  r1);
    vst1q_s32(data + 12, r3);
}

#else

HSORT_ALWAYS_INLINE void cas_neon_8b_soa(int32x4_t& ka, int32x4_t& ida,
                                         int32x4_t& kb, int32x4_t& idb) noexcept {
    const uint32x4_t mask = vcgtq_s32(ka, kb);
    const int32x4_t  k_lo = vbslq_s32(mask, kb, ka);
    const int32x4_t  k_hi = vbslq_s32(mask, ka, kb);
    const int32x4_t  i_lo = vbslq_s32(mask, idb, ida);
    const int32x4_t  i_hi = vbslq_s32(mask, ida, idb);
    ka  = k_lo;  kb  = k_hi;
    ida = i_lo;  idb = i_hi;
}

HSORT_ALWAYS_INLINE void sort_double_leaf_neon_8b(Item* data) noexcept {
    // Hardware de-interleaving load (`ld2`): loads 16 records (2 * LEAF_SIZE = 128B)
    // directly into 4 key registers (k0..k3) and 4 entity_id registers (i0..i3)!
    const int32x4x2_t d0 = vld2q_s32(reinterpret_cast<const std::int32_t*>(data + 0));
    const int32x4x2_t d1 = vld2q_s32(reinterpret_cast<const std::int32_t*>(data + 4));
    const int32x4x2_t d2 = vld2q_s32(reinterpret_cast<const std::int32_t*>(data + 8));
    const int32x4x2_t d3 = vld2q_s32(reinterpret_cast<const std::int32_t*>(data + 12));

    int32x4_t k0 = d0.val[0], i0 = d0.val[1];
    int32x4_t k1 = d1.val[0], i1 = d1.val[1];
    int32x4_t k2 = d2.val[0], i2 = d2.val[1];
    int32x4_t k3 = d3.val[0], i3 = d3.val[1];

    // 4-wire vertical sort across 4 lanes (16 records in parallel, 0 shuffles)
    cas_neon_8b_soa(k0, i0, k1, i1); cas_neon_8b_soa(k2, i2, k3, i3);
    cas_neon_8b_soa(k0, i0, k2, i2); cas_neon_8b_soa(k1, i1, k3, i3);
    cas_neon_8b_soa(k1, i1, k2, i2);

    transpose_4x4_s32(k0, k1, k2, k3);
    transpose_4x4_s32(i0, i1, i2, i3);

    vst2q_s32(reinterpret_cast<std::int32_t*>(data + 0),  int32x4x2_t{{k0, i0}});
    vst2q_s32(reinterpret_cast<std::int32_t*>(data + 4),  int32x4x2_t{{k1, i1}});
    vst2q_s32(reinterpret_cast<std::int32_t*>(data + 8),  int32x4x2_t{{k2, i2}});
    vst2q_s32(reinterpret_cast<std::int32_t*>(data + 12), int32x4x2_t{{k3, i3}});
}

#endif // BENCH_RECORD

#endif // HSORT_IDEA6_ARM64

} // anonymous namespace

void simd_transpose_tiled_merge_sort(Item* data, std::size_t n) {
#if defined(HSORT_IDEA6_X64)
    if (has_avx2()) {
        dispatch_avx2_sort(data, n);
        return;
    }
#elif defined(HSORT_IDEA6_ARM64) && !defined(BENCH_RECORD)
    if (n == LEAF_SIZE) {
        sort_single_leaf_neon(data);
        return;
    }
#endif
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }
    tiled_pingpong_merge_sort(data, n);
}

} // namespace hsort
