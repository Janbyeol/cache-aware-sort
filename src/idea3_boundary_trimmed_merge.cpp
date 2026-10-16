#include "hybrid_sorts.hpp"

namespace hsort {

// ============================================================================
// [Idea 3] 64B 블록 정렬 + 역방향 교차 포인터 경계 트리밍 병합 정렬
//   1. 배열을 64B 캐시 라인 단위(LEAF_SIZE)로 잘라 각각 L1 캐시 내에서 삽입 정렬합니다.
//   2. 인접한 두 정렬 블록을 병합할 때, 중앙 경계(mid)에서
//      - 왼쪽 포인터는 오른쪽 -> 왼쪽 (<-)
//      - 오른쪽 포인터는 왼쪽 -> 오른쪽 (->)
//      반대 방향으로 이동하며 실제로 역전(Inversion)이 발생한 경계 구간만 특정합니다.
//   3. 정렬된 블록 특성상 비교 조건이 단조롭게 유지되므로 CPU 분기 예측률이 극대화됩니다.
// ============================================================================
void boundary_trimmed_merge_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }

    AlignedBuffer buffer(n);

    // 1단계: 64B 캐시 라인 단위로 잘라 삽입 정렬 수행
    for (std::size_t begin = 0; begin < n; begin += LEAF_SIZE) {
        const std::size_t count = std::min<std::size_t>(LEAF_SIZE, n - begin);
        detail::insertion_sort_leaf(data + begin, count);
    }

    // 2단계: 교차 포인터(left <-, right ->)로 역전된 경계만 잘라내어 양방향 병합
    for (std::size_t width = LEAF_SIZE; width < n; width *= 2) {
        for (std::size_t begin = 0; begin + width < n; begin += 2 * width) {
            const std::size_t mid = begin + width;
            const std::size_t end = std::min(n, mid + width);

            const Item left_max  = data[mid - 1];
            const Item right_min = data[mid];

            // 이미 정렬된 경계는 O(1) 비교 한 번으로 통째로 생략
            if (left_max <= right_min) continue;

            // 경계에서 바깥쪽으로 이동하며 역전이 발생한 범위만 특정
            std::size_t left_ptr  = mid;
            std::size_t right_ptr = mid;
            while (left_ptr > begin && data[left_ptr - 1] > right_min) --left_ptr;
            while (right_ptr < end  && data[right_ptr]    < left_max)  ++right_ptr;

            detail::bidirectional_branchless_merge(data, buffer.data(), left_ptr, mid, right_ptr);
            std::memcpy(data + left_ptr, buffer.data() + left_ptr,
                        (right_ptr - left_ptr) * sizeof(Item));
        }
    }
}

} // namespace hsort
