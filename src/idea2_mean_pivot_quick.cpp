#include "hybrid_sorts.hpp"

namespace hsort {

namespace {

// ============================================================================
// [Idea 2] Piggybacked Summation (Mean-Pivot Prediction) Quick Sort
//
// Why this idea?
//   During a Quick Sort partition pass, the CPU already reads every element in the subarray.
//   What if we "piggyback" on this memory scan to accumulate the sum (and min/max bounds)
//   of the left and right partitions in registers?
//   Then, in the next recursive step, we can immediately use the exact arithmetic mean
//   (sum / count) as our next pivot—predicting the center of the distribution without
//   touching memory again.
// ============================================================================
struct SumPartitionResult {
    std::size_t  split;
    std::int64_t left_sum;
    std::int64_t right_sum;
    std::int32_t left_min;
    std::int32_t left_max;
    std::int32_t right_min;
    std::int32_t right_max;
};

HSORT_ALWAYS_INLINE SumPartitionResult sum_partition(Item* __restrict data, std::size_t n, Item pivot) noexcept {
    std::ptrdiff_t left  = -1;
    std::ptrdiff_t right = static_cast<std::ptrdiff_t>(n);

    // 1. Standard Hoare partition to divide [0, split) and [split, n)
    for (;;) {
        do { ++left;  } while (data[left]  < pivot);
        do { --right; } while (data[right] > pivot);
        if (left >= right) break;
        std::swap(data[left], data[right]);
    }

    const std::size_t split = static_cast<std::size_t>(right + 1);

    // 2. Accumulate sum and min/max bounds for both partitions while data is hot in cache
    std::int64_t left_sum  = 0;
    std::int32_t left_min  = get_key(data[0]);
    std::int32_t left_max  = left_min;
    for (std::size_t i = 0; i < split; ++i) {
        const std::int32_t k = get_key(data[i]);
        left_sum += k;
        if (k < left_min) left_min = k;
        if (k > left_max) left_max = k;
    }

    std::int64_t right_sum = 0;
    std::int32_t right_min = (split < n) ? get_key(data[split]) : 0;
    std::int32_t right_max = right_min;
    for (std::size_t i = split; i < n; ++i) {
        const std::int32_t k = get_key(data[i]);
        right_sum += k;
        if (k < right_min) right_min = k;
        if (k > right_max) right_max = k;
    }

    return {split, left_sum, right_sum, left_min, left_max, right_min, right_max};
}

// Computes the mean pivot safely.
// Integer division truncates toward zero, which on negative numbers (e.g., nineteen -5s and one -4)
// can yield mean == max_key (-99 / 20 == -4). In Hoare partitioning, a pivot equal to max_key
// can send all elements to the left partition (split == count), causing infinite recursion.
// Clamping the mean to [min_key, max_key - 1] guarantees forward progress on every step.
HSORT_ALWAYS_INLINE Item make_safe_mean_pivot(std::int64_t sum, std::size_t count,
                                              std::int32_t min_key, std::int32_t max_key) noexcept {
    std::int64_t mean = sum / static_cast<std::int64_t>(count);
    if (mean >= max_key) mean = static_cast<std::int64_t>(max_key) - 1;
    if (mean <  min_key) mean = static_cast<std::int64_t>(min_key);
    return make_item(static_cast<std::int32_t>(mean));
}

void mean_quick_sort_impl(Item* data, std::size_t n, Item pivot) noexcept {
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }

    const SumPartitionResult part = sum_partition(data, n, pivot);

    // Sort left partition (skip immediately if all keys are identical: left_min == left_max)
    if (part.split <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, part.split);
    } else if (part.left_min < part.left_max) {
        const Item next_left_pivot = make_safe_mean_pivot(
            part.left_sum, part.split, part.left_min, part.left_max
        );
        mean_quick_sort_impl(data, part.split, next_left_pivot);
    }

    // Sort right partition (skip immediately if all keys are identical: right_min == right_max)
    const std::size_t right_count = n - part.split;
    if (right_count <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data + part.split, right_count);
    } else if (part.right_min < part.right_max) {
        const Item next_right_pivot = make_safe_mean_pivot(
            part.right_sum, right_count, part.right_min, part.right_max
        );
        mean_quick_sort_impl(data + part.split, right_count, next_right_pivot);
    }
}

} // anonymous namespace

void mean_pivot_quick_sort(Item* data, std::size_t n) noexcept {
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }
    const Item initial_pivot = detail::median_of_three(data[0], data[(n - 1) / 2], data[n - 1]);
    mean_quick_sort_impl(data, n, initial_pivot);
}

} // namespace hsort
