#include "hybrid_sorts.hpp"

namespace hsort {

namespace {

// Find the first index in sorted [begin, mid) where data[idx] > right_min.
// Starts at `mid` and walks left (<-):
//   1. Fast linear scan for up to 8 elements (residing in the same 64B cache line as `mid`).
//   2. O(1) outer-boundary check (`data[begin] > right_min`) to skip full-block scans on random data.
//   3. Exponential galloping (step = 8, 16, 32...) + binary search to locate the exact boundary
//      in O(log K) without polluting L1 cache lines.
HSORT_ALWAYS_INLINE std::size_t trim_left_boundary(const Item* __restrict data,
                                                   std::size_t begin,
                                                   std::size_t mid,
                                                   Item right_min) noexcept {
    std::size_t left_ptr = mid;
    const std::size_t linear_limit = (mid - begin > 8) ? (mid - 8) : begin;

    while (left_ptr > linear_limit && data[left_ptr - 1] > right_min) {
        --left_ptr;
    }
    if (left_ptr > linear_limit || left_ptr == begin || !(data[left_ptr - 1] > right_min)) {
        return left_ptr;
    }

    // O(1) check: does the entire left block need merging?
    if (data[begin] > right_min) {
        return begin;
    }

    // Exponential galloping from (left_ptr - 1) towards begin
    std::size_t hi   = left_ptr - 1; // invariant: data[hi] > right_min
    std::size_t lo   = begin + 1;    // invariant: data[begin] <= right_min
    std::size_t step = 8;

    while (hi - lo > step) {
        const std::size_t probe = hi - step;
        if (data[probe] > right_min) {
            hi = probe;
            step <<= 1;
        } else {
            lo = probe + 1;
            break;
        }
    }

    while (lo < hi) {
        const std::size_t m = lo + ((hi - lo) >> 1);
        if (data[m] > right_min) hi = m;
        else                     lo = m + 1;
    }
    return lo;
}

// Find the first index in sorted [mid, end) where !(data[idx] < left_max).
// Starts at `mid` and walks right (->):
//   1. Fast linear scan for up to 8 elements (same 64B cache line as `mid`).
//   2. O(1) outer-boundary check (`data[end - 1] < left_max`).
//   3. Exponential galloping + binary search in O(log K).
HSORT_ALWAYS_INLINE std::size_t trim_right_boundary(const Item* __restrict data,
                                                    std::size_t mid,
                                                    std::size_t end,
                                                    Item left_max) noexcept {
    std::size_t right_ptr = mid;
    const std::size_t linear_limit = (end - mid > 8) ? (mid + 8) : end;

    while (right_ptr < linear_limit && data[right_ptr] < left_max) {
        ++right_ptr;
    }
    if (right_ptr < linear_limit || right_ptr == end || !(data[right_ptr] < left_max)) {
        return right_ptr;
    }

    // O(1) check: does the entire right block need merging?
    if (data[end - 1] < left_max) {
        return end;
    }

    // Exponential galloping from right_ptr towards (end - 1)
    std::size_t lo   = right_ptr + 1;
    std::size_t hi   = end - 1; // invariant: !(data[hi] < left_max)
    std::size_t step = 8;

    while (hi - lo > step) {
        const std::size_t probe = lo + step;
        if (data[probe] < left_max) {
            lo = probe + 1;
            step <<= 1;
        } else {
            hi = probe;
            break;
        }
    }

    while (lo < hi) {
        const std::size_t m = lo + ((hi - lo) >> 1);
        if (data[m] < left_max) lo = m + 1;
        else                    hi = m;
    }
    return lo;
}

// Bidirectional branchless merge writing into `dst[0 .. end - begin)` (anchored at offset 0 of scratch buffer)
// so small and medium merges always hit hot L1/L2 cache lines at the start of `buffer`.
HSORT_ALWAYS_INLINE void bidirectional_merge_to_scratch(const Item* __restrict src,
                                                        Item* __restrict scratch,
                                                        std::size_t begin,
                                                        std::size_t mid,
                                                        std::size_t end) noexcept {
    const std::size_t total = end - begin;
    std::size_t l_fwd   = begin, l_bwd   = mid;
    std::size_t r_fwd   = mid,   r_bwd   = end;
    std::size_t out_fwd = 0,     out_bwd = total;

    while (l_fwd < l_bwd && r_fwd < r_bwd) {
        const std::size_t safe_pairs = std::min(l_bwd - l_fwd, r_bwd - r_fwd);

        for (std::size_t s = 0; s < safe_pairs; ++s) {
            const Item lv         = src[l_fwd];
            const Item rv         = src[r_fwd];
            const bool take_right = (rv < lv);
            scratch[out_fwd++]    = take_right ? rv : lv;
            l_fwd                += !take_right;
            r_fwd                += take_right;

            const Item lb        = src[l_bwd - 1];
            const Item rb        = src[r_bwd - 1];
            const bool take_left = (lb > rb);
            scratch[--out_bwd]   = take_left ? lb : rb;
            l_bwd               -= take_left;
            r_bwd               -= !take_left;
        }
    }

    if (l_fwd < l_bwd) {
        std::memcpy(scratch + out_fwd, src + l_fwd, (l_bwd - l_fwd) * sizeof(Item));
    } else if (r_fwd < r_bwd) {
        std::memcpy(scratch + out_fwd, src + r_fwd, (r_bwd - r_fwd) * sizeof(Item));
    }
}

// Half-buffer in-place merge: copies ONLY the smaller side into `scratch[0 .. min(L, R))`
// and merges directly into `data[left_ptr .. right_ptr)`, eliminating the post-merge memcpy
// and leaving any unconsumed tail of the larger side untouched in place.
HSORT_ALWAYS_INLINE void half_buffer_merge_left_smaller(Item* __restrict data,
                                                        Item* __restrict scratch,
                                                        std::size_t left_ptr,
                                                        std::size_t mid,
                                                        std::size_t right_ptr) noexcept {
    const std::size_t left_len = mid - left_ptr;
    std::memcpy(scratch, data + left_ptr, left_len * sizeof(Item));

    std::size_t i   = 0;
    std::size_t j   = mid;
    std::size_t out = left_ptr;

    while (i < left_len && j < right_ptr) {
        const Item lv         = scratch[i];
        const Item rv         = data[j];
        const bool take_right = (rv < lv);
        data[out++]           = take_right ? rv : lv;
        i                    += !take_right;
        j                    += take_right;
    }
    if (i < left_len) {
        std::memcpy(data + out, scratch + i, (left_len - i) * sizeof(Item));
    }
}

HSORT_ALWAYS_INLINE void half_buffer_merge_right_smaller(Item* __restrict data,
                                                         Item* __restrict scratch,
                                                         std::size_t left_ptr,
                                                         std::size_t mid,
                                                         std::size_t right_ptr) noexcept {
    const std::size_t right_len = right_ptr - mid;
    std::memcpy(scratch, data + mid, right_len * sizeof(Item));

    std::size_t i   = mid;
    std::size_t j   = right_len;
    std::size_t out = right_ptr;

    while (i > left_ptr && j > 0) {
        const Item lb        = data[i - 1];
        const Item rb        = scratch[j - 1];
        const bool take_left = (lb > rb);
        data[--out]          = take_left ? lb : rb;
        i                   -= take_left;
        j                   -= !take_left;
    }
    if (j > 0) {
        std::memcpy(data + left_ptr, scratch, j * sizeof(Item));
    }
}

HSORT_ALWAYS_INLINE void trimmed_merge_pair(Item* __restrict data,
                                            Item* __restrict scratch,
                                            std::size_t begin,
                                            std::size_t mid,
                                            std::size_t end) noexcept {
    const Item left_max  = data[mid - 1];
    const Item right_min = data[mid];

    // O(1) fast-path: skip the entire merge if the boundary is already ordered
    if (left_max <= right_min) return;

    const std::size_t left_ptr  = trim_left_boundary(data, begin, mid, right_min);
    const std::size_t right_ptr = trim_right_boundary(data, mid, end, left_max);

    const std::size_t left_len  = mid - left_ptr;
    const std::size_t right_len = right_ptr - mid;
    const std::size_t total_len = right_ptr - left_ptr;

    // When the overlap is tiny or asymmetric (e.g., Append 5% Tail), copy only the smaller half
    // into `scratch[0]` and merge directly into `data` (cuts memory writes in half and avoids full memcpy back).
    if (left_len <= right_len) {
        if (left_len <= 8 || (left_len * 4 <= right_len)) {
            half_buffer_merge_left_smaller(data, scratch, left_ptr, mid, right_ptr);
            return;
        }
    } else {
        if (right_len <= 8 || (right_len * 4 <= left_len)) {
            half_buffer_merge_right_smaller(data, scratch, left_ptr, mid, right_ptr);
            return;
        }
    }

    // For balanced large overlaps, use high-ILP bidirectional branchless merge into `scratch[0 .. total_len)`
    bidirectional_merge_to_scratch(data, scratch, left_ptr, mid, right_ptr);
    std::memcpy(data + left_ptr, scratch, total_len * sizeof(Item));
}

} // anonymous namespace

// ============================================================================
// [Idea 3] 64-Byte Micro-Block Sort + Cross-Pointer Boundary-Trimmed Merge Sort
//
// Key Optimizations for Cache Hit Rate & Reduced Memory Copy:
//   1. L1-Windowed Bottom-Up Execution:
//      We still slice the array into 64-byte cache-line blocks (LEAF_SIZE), but instead
//      of sweeping the entire multi-megabyte array at each micro-width (which evicts L1D),
//      we complete the 64B leaf sort and early boundary-trimmed merges (width < TILE_SIZE)
//      within each 8KiB L1 window while the cache lines are hot in L1D.
//   2. Hybrid Galloping Cross-Pointer Trimming:
//      Starts at `mid` and walks outward in opposite directions (left <- mid -> right):
//      scans up to 8 items linearly (inside `mid`'s cache line), then uses O(1) outer-bound
//      checks and exponential galloping to avoid scanning thousands of cache lines on random data.
//   3. Anchored Scratch (`scratch[0]`) + Half-Buffer In-Place Merge:
//      Small/asymmetric overlaps copy ONLY `min(left_len, right_len)` to `scratch[0]` and
//      merge directly into `data` (eliminating the post-merge memcpy). Balanced overlaps
//      merge into `scratch[0 .. len)` so the scratch buffer stays pinned in L1/L2 cache.
// ============================================================================
void boundary_trimmed_merge_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }

    AlignedBuffer buffer(n);
    Item* const scratch = buffer.data();

    // Phase 1: Process 64B leaf sorts and early boundary-trimmed merges inside 8KiB L1 windows
    for (std::size_t tile_begin = 0; tile_begin < n; tile_begin += TILE_SIZE) {
        const std::size_t tile_end = std::min(n, tile_begin + TILE_SIZE);

        for (std::size_t begin = tile_begin; begin < tile_end; begin += LEAF_SIZE) {
            const std::size_t count = std::min<std::size_t>(LEAF_SIZE, tile_end - begin);
            detail::insertion_sort_leaf(data + begin, count);
        }

        for (std::size_t width = LEAF_SIZE; width < (tile_end - tile_begin); width *= 2) {
            for (std::size_t begin = tile_begin; begin + width < tile_end; begin += 2 * width) {
                const std::size_t mid = begin + width;
                const std::size_t end = std::min(tile_end, mid + width);
                trimmed_merge_pair(data, scratch, begin, mid, end);
            }
        }
    }

    // Phase 2: Global boundary-trimmed merges across L1 windows (width >= TILE_SIZE)
    for (std::size_t width = TILE_SIZE; width < n; width *= 2) {
        for (std::size_t begin = 0; begin + width < n; begin += 2 * width) {
            const std::size_t mid = begin + width;
            const std::size_t end = std::min(n, mid + width);
            trimmed_merge_pair(data, scratch, begin, mid, end);
        }
    }
}

} // namespace hsort
