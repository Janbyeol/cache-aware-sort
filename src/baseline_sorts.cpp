#include "hybrid_sorts.hpp"

namespace hsort {

// [Baseline 1] 순수 재귀 호어 퀵 정렬 (크기 1까지 끝까지 분할하여 캐시 컷오프 미적용)
void pure_quick_sort(Item* data, std::size_t n) noexcept {
    detail::quick_sort_impl<1>(data, n);
}

// [Baseline 2] C++ 표준 std::sort (Introsort)
void std_sort_wrapper(Item* data, std::size_t n) noexcept {
    std::sort(data, data + n);
}

// [Baseline 3] C++ 표준 std::stable_sort (Merge Sort 계열)
void std_stable_sort_wrapper(Item* data, std::size_t n) {
    std::stable_sort(data, data + n);
}

} // namespace hsort
