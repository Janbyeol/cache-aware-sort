#pragma once
#include "sort_types.hpp"

namespace hsort {

// [Baseline 1] 순수 재귀 호어 퀵 정렬 (크기 1까지 끝까지 분할)
void pure_quick_sort(Item* data, std::size_t n) noexcept;

// [Baseline 2] C++ 표준 std::sort (Introsort)
void std_sort_wrapper(Item* data, std::size_t n) noexcept;

// [Baseline 3] C++ 표준 std::stable_sort (Merge Sort 계열)
void std_stable_sort_wrapper(Item* data, std::size_t n);

// [Idea 1] 64B 캐시 라인 리프 제한 + 선형 삽입 하이브리드 퀵 정렬
void cache_leaf_quick_sort(Item* data, std::size_t n) noexcept;

// [Idea 2] 무임승차 합산(Summation) 평균 피벗 예측 퀵 정렬
void mean_pivot_quick_sort(Item* data, std::size_t n);

// [Idea 3] 64B 블록 정렬 + 역방향 교차 포인터 경계 트리밍 병합 정렬
void boundary_trimmed_merge_sort(Item* data, std::size_t n);

// [Idea 4: 최종 융합] 거시적 경향 감지 + L1 타일링 + 이진 트리밍 + 핑퐁 병합 정렬
void tiled_pingpong_merge_sort(Item* data, std::size_t n);

} // namespace hsort