#include "hybrid_sorts.hpp"

namespace hsort {

// ============================================================================
// [Idea 1] 64B 캐시 라인 리프 제한 + 선형 삽입 하이브리드 퀵 정렬
//   - 배열을 끝까지(크기 1) 재귀 분할하지 않고, 정확히 64B 캐시 라인(LEAF_SIZE) 이하로
//     내려가면 재귀를 멈춘 뒤 L1 캐시 내부에서 명령어가 최소화된 선형 삽입 정렬로 마무리합니다.
// ============================================================================
void cache_leaf_quick_sort(Item* data, std::size_t n) noexcept {
    detail::quick_sort_impl<LEAF_SIZE>(data, n);
}

} // namespace hsort
