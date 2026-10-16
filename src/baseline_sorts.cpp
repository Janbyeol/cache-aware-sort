#include "hybrid_sorts.hpp"

namespace hsort {

// [Baseline 1] Pure Recursive Hoare Quick Sort
// Recurses all the way down to size 1 without stopping at the 64B cache-line boundary.
// Used to measure how much overhead we save in Idea 1 by stopping at 64B leaves.
void pure_quick_sort(Item* data, std::size_t n) noexcept {
    detail::quick_sort_impl<1>(data, n);
}

// [Baseline 2] C++ Standard Unstable Sort (Introsort / pdqsort)
void std_sort_wrapper(Item* data, std::size_t n) noexcept {
    std::sort(data, data + n);
}

// [Baseline 3] C++ Standard Stable Sort (Merge Sort variant)
void std_stable_sort_wrapper(Item* data, std::size_t n) {
    std::stable_sort(data, data + n);
}

} // namespace hsort
