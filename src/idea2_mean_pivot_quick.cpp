#include "hybrid_sorts.hpp"

namespace hsort {

namespace {

// ============================================================================
// [Idea 2] 무임승차 합산(Piggybacked Summation) 평균 피벗 예측 퀵 정렬
//   - 분할(Partition) 과정에서 어차피 모든 원소를 읽으므로, 좌/우 파티션으로 분류되는
//     원소들의 합(Sum)과 최솟값/최댓값을 레지스터에서 함께 누적합니다.
//   - 다음 재귀 단계에서는 추가 메모리 접근 없이 계산된 평균(Mean)을 피벗으로 사용합니다.
//   - [Bug Fix] 정수 나눗셈 버림이나 이상치(Outlier)로 인해 평균 피벗이 구간의
//     최솟값/최댓값 경계에 걸려 한쪽 파티션이 0개가 되는 무한 재귀 버그를 완벽히 방지합니다.
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

    // 고전 Hoare 분할로 안전하게 좌/우 파티션 경계(split) 확정
    for (;;) {
        do { ++left;  } while (data[left]  < pivot);
        do { --right; } while (data[right] > pivot);
        if (left >= right) break;
        std::swap(data[left], data[right]);
    }

    const std::size_t split = static_cast<std::size_t>(right + 1);

    // L1 캐시에 방금 로드된 좌/우 구간의 합계 및 최대/최소 경계 산출
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

// 구간의 평균값을 계산하되, Hoare 분할이 절대 한쪽(0 : N)으로 쏠리지 않도록
// [min_key, max_key - 1] 범위 안으로 안전하게 클램핑한 피벗을 반환합니다.
HSORT_ALWAYS_INLINE Item make_safe_mean_pivot(std::int64_t sum, std::size_t count,
                                              std::int32_t min_key, std::int32_t max_key) noexcept {
    std::int64_t mean = sum / static_cast<std::int64_t>(count);
    // 음수 나눗셈 등에서 mean이 max_key 이상이 되면 data[right] > pivot 조건이 모두 거짓이 되어
    // split == count가 되는 무한 재귀가 발생할 수 있으므로 상한을 max_key - 1로 제한
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

    // 1. 좌측 부분 배열 정렬 (모든 키가 동일하면 이미 정렬된 상태이므로 즉시 종료)
    if (part.split <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, part.split);
    } else if (part.left_min < part.left_max) {
        const Item next_left_pivot = make_safe_mean_pivot(
            part.left_sum, part.split, part.left_min, part.left_max
        );
        mean_quick_sort_impl(data, part.split, next_left_pivot);
    }

    // 2. 우측 부분 배열 정렬 (모든 키가 동일하면 이미 정렬된 상태이므로 즉시 종료)
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
