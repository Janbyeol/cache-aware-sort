#include "hybrid_sorts.hpp"

namespace hsort {

// ============================================================================
// [Idea 3] 64-Byte Micro-Block Sort + Cross-Pointer Boundary-Trimmed Merge Sort
//
// Why this idea?
//   1. Bottom-Up 64B Blocks: Instead of top-down Quick Sort recursion, we slice the array
//      directly into 64-byte cache-line blocks (LEAF_SIZE) and sort each block in L1 cache.
//   2. Cross-Pointer Boundary Trimming: When merging two adjacent sorted blocks [begin, mid)
//      and [mid, end), we start two pointers at the center boundary (`mid`) and walk them
//      in opposite directions:
//        - `left_ptr`  walks right-to-left (<-) while elements are > right_min
//        - `right_ptr` walks left-to-right (->) while elements are < left_max
//   3. Why is this fast?
//      Because both blocks are already sorted, the loop condition evaluates to `true`
//      continuously and flips to `false` exactly once at the boundary. The CPU branch
//      predictor achieves near-100% accuracy, and already-sorted outer regions are
//      completely untouched in memory (zero copy overhead on sorted prefixes/suffixes).
// ============================================================================
void boundary_trimmed_merge_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }

    AlignedBuffer buffer(n);

    // Step 1: Sort each 64-byte cache-line block in place
    for (std::size_t begin = 0; begin < n; begin += LEAF_SIZE) {
        const std::size_t count = std::min<std::size_t>(LEAF_SIZE, n - begin);
        detail::insertion_sort_leaf(data + begin, count);
    }

    // Step 2: Bottom-up merge with opposite-direction boundary trimming
    for (std::size_t width = LEAF_SIZE; width < n; width *= 2) {
        for (std::size_t begin = 0; begin + width < n; begin += 2 * width) {
            const std::size_t mid = begin + width;
            const std::size_t end = std::min(n, mid + width);

            const Item left_max  = data[mid - 1];
            const Item right_min = data[mid];

            // O(1) fast-path: skip the entire merge if the boundary is already ordered
            if (left_max <= right_min) continue;

            // Walk outward from `mid` in opposite directions (left <- mid -> right)
            // to isolate only the inverted subrange [left_ptr, right_ptr)
            std::size_t left_ptr  = mid;
            std::size_t right_ptr = mid;
            while (left_ptr > begin && data[left_ptr - 1] > right_min) --left_ptr;
            while (right_ptr < end  && data[right_ptr]    < left_max)  ++right_ptr;

            // Merge only the trimmed overlap window and copy back
            detail::bidirectional_branchless_merge(data, buffer.data(), left_ptr, mid, right_ptr);
            std::memcpy(data + left_ptr, buffer.data() + left_ptr,
                        (right_ptr - left_ptr) * sizeof(Item));
        }
    }
}

} // namespace hsort
