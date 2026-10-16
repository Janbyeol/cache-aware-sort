#include "hybrid_sorts.hpp"

namespace hsort {

// ============================================================================
// [Idea 1] 64-Byte Cache-Line Cutoff + Linear Insertion Quick Sort
//
// Why this idea?
//   A standard CPU fetches memory in 64-byte cache lines (16x int32_t or 8x 8B records).
//   Once a subarray is small enough to fit inside a single 64-byte cache line (LEAF_SIZE),
//   partitioning it further with Quick Sort gives zero cache-hit advantage and only adds
//   recursion and branch misprediction overhead.
//   So we stop recursion at LEAF_SIZE and finish each micro-block in L1 cache using
//   a tight linear insertion sort.
// ============================================================================
void cache_leaf_quick_sort(Item* data, std::size_t n) noexcept {
    detail::quick_sort_impl<LEAF_SIZE>(data, n);
}

} // namespace hsort
