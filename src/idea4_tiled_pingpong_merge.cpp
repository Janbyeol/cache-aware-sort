#include "hybrid_sorts.hpp"

namespace hsort {

namespace {

// ============================================================================
// [Idea 4] Global Trend Probe + 8KiB L1 Tiling + Ping-Pong Merge Sort
//
// How this evolved from Idea 3:
//   While Idea 3 excels on nearly-sorted data, profiling large random arrays (N = 1,000,000)
//   revealed two bottlenecks:
//     1. Cache Eviction: Sorting all 64B blocks across the entire array before the first
//        merge pass evicts earlier blocks from the L1 cache.
//     2. Copy Overhead: Copying merged elements back to `data` via `memcpy` on every level
//        wastes memory bandwidth when the overlap window is large.
//
//   Idea 4 solves both problems:
//     - L1 Cache Tiling (8 KiB): We process the array in 8 KiB tiles (TILE_SIZE) so the
//       entire tile and its auxiliary buffer stay hot in L1D cache across local merge passes.
//     - Ping-Pong Merging: Instead of copying back to `data` after every pass, we alternate
//       the source and destination roles between `data` and `buffer`.
//     - Binary Boundary Trimming: For larger blocks, linear boundary scanning is replaced
//       with `std::upper_bound` / `std::lower_bound` to find the overlap window in O(log W).
// ============================================================================

// Lightweight O(1) probe (32 equidistant samples) to detect globally ascending or
// descending arrays before allocating the auxiliary buffer.
inline bool probe_and_handle_global_trend(Item* data, std::size_t n) noexcept {
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

    // If the array is overwhelmingly descending, flip inverted pairs from both ends
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

inline void pingpong_merge_pair(const Item* __restrict src, Item* __restrict dst,
                                std::size_t begin, std::size_t mid, std::size_t end) noexcept {
    // Fast-path: if already ordered across the boundary, bulk-copy to the ping-pong target
    if (mid >= end || src[mid - 1] <= src[mid]) {
        std::memcpy(dst + begin, src + begin, (end - begin) * sizeof(Item));
        return;
    }

    // Binary search for the exact inverted boundary window [trim_left, trim_right)
    const std::size_t trim_left = static_cast<std::size_t>(
        std::upper_bound(src + begin, src + mid, src[mid]) - src
    );
    const std::size_t trim_right = static_cast<std::size_t>(
        std::lower_bound(src + mid, src + end, src[mid - 1]) - src
    );

    // Bulk-copy the non-overlapping outer regions directly
    if (trim_left > begin) {
        std::memcpy(dst + begin, src + begin, (trim_left - begin) * sizeof(Item));
    }
    if (trim_right < end) {
        std::memcpy(dst + trim_right, src + trim_right, (end - trim_right) * sizeof(Item));
    }

    // Merge only the overlapping interior window
    detail::bidirectional_branchless_merge(src, dst, trim_left, mid, trim_right);
}

inline void pingpong_merge_pass(const Item* __restrict src, Item* __restrict dst,
                                std::size_t begin, std::size_t end, std::size_t width) noexcept {
    for (std::size_t first = begin; first < end; first += 2 * width) {
        const std::size_t mid  = std::min(end, first + width);
        const std::size_t last = std::min(end, mid + width);
        pingpong_merge_pair(src, dst, first, mid, last);
    }
}

} // anonymous namespace

void tiled_pingpong_merge_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }

    // Step 1: Check for global ascending/descending trends
    if (probe_and_handle_global_trend(data, n)) return;

    AlignedBuffer buffer(n);

    // Step 2: L1 Cache Tiling (8 KiB per tile)
    // Sort 64B leaves and complete all local ping-pong merges while the tile is hot in L1D.
    for (std::size_t tile_begin = 0; tile_begin < n; tile_begin += TILE_SIZE) {
        const std::size_t tile_end = std::min(n, tile_begin + TILE_SIZE);

        for (std::size_t first = tile_begin; first < tile_end; first += LEAF_SIZE) {
            const std::size_t last = std::min(tile_end, first + LEAF_SIZE);

            // Reverse any strictly descending prefix inside the 64B leaf before insertion sort
            std::size_t desc_end = first + 1;
            while (desc_end < last && data[desc_end] < data[desc_end - 1]) {
                ++desc_end;
            }
            if (desc_end - first > 1) {
                std::reverse(data + first, data + desc_end);
            }
            detail::insertion_sort_leaf(data + first, last - first);
        }

        Item* src = data;
        Item* dst = buffer.data();
        for (std::size_t width = LEAF_SIZE; width < tile_end - tile_begin; width *= 2) {
            pingpong_merge_pass(src, dst, tile_begin, tile_end, width);
            std::swap(src, dst);
        }

        // Ensure every finished tile lands in `buffer` so global merge starts from a uniform source
        if (src != buffer.data()) {
            std::memcpy(buffer.data() + tile_begin, src + tile_begin,
                        (tile_end - tile_begin) * sizeof(Item));
        }
    }

    // Step 3: Global Ping-Pong Merge across sorted L1 tiles
    Item* src = buffer.data();
    Item* dst = data;
    for (std::size_t width = TILE_SIZE; width < n; width *= 2) {
        pingpong_merge_pass(src, dst, 0, n, width);
        std::swap(src, dst);
    }

    if (src != data) {
        std::memcpy(data, src, n * sizeof(Item));
    }
}

} // namespace hsort
