#include "hybrid_sorts.hpp"
#include <algorithm>
#include <cstring>
#include <utility>

namespace hsort {

namespace {

// ============================================================================
// 내부 헬퍼 함수 (Internal Helpers)
// ============================================================================

[[gnu::always_inline]] inline Item median_of_three(Item a, Item b, Item c) noexcept {
    if (a < b) return (b < c) ? b : std::max(a, c);
    return (a < c) ? a : std::max(b, c);
}

// 64B 캐시 라인(16개 또는 8개 원소) 내부를 처리하는 고속 선형 삽입 정렬
[[gnu::always_inline]] inline void insertion_sort_leaf(Item* __restrict data, std::size_t n) noexcept {
    for (std::size_t i = 1; i < n; ++i) {
        const Item key = data[i];
        std::size_t j  = i;
        while (j > 0 && key < data[j - 1]) {
            data[j] = data[j - 1];
            --j;
        }
        data[j] = key;
    }
}

// 고전적인 호어 분할 (Hoare Partition)
inline std::size_t hoare_partition(Item* __restrict data, std::size_t n, Item pivot) noexcept {
    std::ptrdiff_t left  = -1;
    std::ptrdiff_t right = static_cast<std::ptrdiff_t>(n);

    for (;;) {
        do { ++left;  } while (data[left]  < pivot);
        do { --right; } while (data[right] > pivot);
        if (left >= right) {
            return static_cast<std::size_t>(right + 1);
        }
        std::swap(data[left], data[right]);
    }
}

template <std::size_t Cutoff>
void quick_sort_impl(Item* data, std::size_t n) noexcept {
    if (n <= Cutoff) {
        if constexpr (Cutoff > 1) {
            insertion_sort_leaf(data, n);
        }
        return;
    }

    const Item pivot        = median_of_three(data[0], data[(n - 1) / 2], data[n - 1]);
    const std::size_t split = hoare_partition(data, n, pivot);

    quick_sort_impl<Cutoff>(data, split);
    quick_sort_impl<Cutoff>(data + split, n - split);
}

// ============================================================================
// [Idea 2 전용 헬퍼] 무임승차 합산 분할 (Piggybacked Summation Partition)
// ============================================================================
struct SumPartitionResult {
    std::size_t split;
    std::int64_t left_sum;
    std::int64_t right_sum;
};

inline SumPartitionResult sum_partition(Item* __restrict data, std::size_t n, Item pivot) noexcept {
    std::size_t  left      = 0;
    std::size_t  right     = n;
    std::int64_t left_sum  = 0;
    std::int64_t right_sum = 0;

    for (;;) {
        while (left < right && data[left] < pivot) {
            left_sum += get_key(data[left++]);
        }
        while (left < right && data[right - 1] > pivot) {
            right_sum += get_key(data[--right]);
        }
        if (left == right) break;

        // 마지막 1개 원소가 피벗과 같을 때 한쪽 파티션이 비지 않도록 분배
        if (right - left == 1) {
            if (left == n - 1) right_sum += get_key(data[--right]);
            else               left_sum  += get_key(data[left++]);
            break;
        }

        const Item lv = data[left];
        const Item rv = data[right - 1];
        data[left++]  = rv;
        data[--right] = lv;

        left_sum  += get_key(rv);
        right_sum += get_key(lv);
    }
    return {left, left_sum, right_sum};
}

void mean_quick_sort_impl(Item* data, std::size_t n, Item pivot) {
    if (n <= LEAF_SIZE) {
        insertion_sort_leaf(data, n);
        return;
    }

    const SumPartitionResult part = sum_partition(data, n, pivot);

    // 좌측 부분 배열 정렬 (누적된 좌측 평균을 다음 피벗으로 즉시 사용)
    if (part.split <= LEAF_SIZE) {
        insertion_sort_leaf(data, part.split);
    } else {
        const auto next_left_key = static_cast<std::int32_t>(
            part.left_sum / static_cast<std::int64_t>(part.split)
        );
        mean_quick_sort_impl(data, part.split, make_item(next_left_key));
    }

    // 우측 부분 배열 정렬 (누적된 우측 평균을 다음 피벗으로 즉시 사용)
    const std::size_t right_count = n - part.split;
    if (right_count <= LEAF_SIZE) {
        insertion_sort_leaf(data + part.split, right_count);
    } else {
        const auto next_right_key = static_cast<std::int32_t>(
            part.right_sum / static_cast<std::int64_t>(right_count)
        );
        mean_quick_sort_impl(data + part.split, right_count, make_item(next_right_key));
    }
}

// ============================================================================
// [Idea 3 & 4 공통 커널] 양방향 분기 제거 병합 (Bidirectional Branchless Merge)
//   - 전면 포인터(최솟값 ->)와 후면 포인터(최댓값 <-)를 동시에 이동시켜 ILP 2배 확보
// ============================================================================
inline void bidirectional_branchless_merge(const Item* __restrict src, Item* __restrict dst,
                                           std::size_t begin, std::size_t mid, std::size_t end) noexcept {
    std::size_t l_fwd   = begin, l_bwd   = mid;
    std::size_t r_fwd   = mid,   r_bwd   = end;
    std::size_t out_fwd = begin, out_bwd = end;

    while (l_fwd < l_bwd && r_fwd < r_bwd) {
        // 더 짧은 쪽의 남은 개수만큼은 루프 중간 탈출 검사 없이 완벽히 병렬 실행
        const std::size_t safe_pairs = std::min(l_bwd - l_fwd, r_bwd - r_fwd);

        for (std::size_t s = 0; s < safe_pairs; ++s) {
            // 1. 앞쪽에서 최솟값 선택 (cmov 명령어 유도)
            const Item lv         = src[l_fwd];
            const Item rv         = src[r_fwd];
            const bool take_right = (rv < lv);
            dst[out_fwd++]        = take_right ? rv : lv;
            l_fwd                += !take_right;
            r_fwd                += take_right;

            // 2. 뒤쪽에서 최댓값 선택 (앞쪽 루프와 데이터 의존성 0 -> 단일 사이클 동시 실행)
            const Item lb        = src[l_bwd - 1];
            const Item rb        = src[r_bwd - 1];
            const bool take_left = (lb > rb);
            dst[--out_bwd]       = take_left ? lb : rb;
            l_bwd               -= take_left;
            r_bwd               -= !take_left;
        }
    }

    // 중앙에 남은 잔여 구간 일괄 복사
    if (l_fwd < l_bwd) {
        std::memcpy(dst + out_fwd, src + l_fwd, (l_bwd - l_fwd) * sizeof(Item));
    } else if (r_fwd < r_bwd) {
        std::memcpy(dst + out_fwd, src + r_fwd, (r_bwd - r_fwd) * sizeof(Item));
    }
}

// ============================================================================
// [Idea 4 전용 헬퍼] 거시적 경향 감지 및 이진 트리밍 패스
// ============================================================================
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

    // 거의 역순인 배열 감지 시 양 끝 대칭 스왑 패스 수행
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
    if (mid >= end || src[mid - 1] <= src[mid]) {
        std::memcpy(dst + begin, src + begin, (end - begin) * sizeof(Item));
        return;
    }

    // 이진 탐색으로 실제 겹치는 경계 구간 [trim_left, trim_right) 산출
    const std::size_t trim_left = std::upper_bound(
        src + begin, src + mid, src[mid]) - src;
    const std::size_t trim_right = std::lower_bound(
        src + mid, src + end, src[mid - 1]) - src;

    if (trim_left > begin) {
        std::memcpy(dst + begin, src + begin, (trim_left - begin) * sizeof(Item));
    }
    if (trim_right < end) {
        std::memcpy(dst + trim_right, src + trim_right, (end - trim_right) * sizeof(Item));
    }

    bidirectional_branchless_merge(src, dst, trim_left, mid, trim_right);
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

// ============================================================================
// 공개 API 구현부 (Public Sorting APIs)
// ============================================================================

void pure_quick_sort(Item* data, std::size_t n) noexcept {
    quick_sort_impl<1>(data, n);
}

void std_sort_wrapper(Item* data, std::size_t n) noexcept {
    std::sort(data, data + n);
}

void std_stable_sort_wrapper(Item* data, std::size_t n) {
    std::stable_sort(data, data + n);
}

void cache_leaf_quick_sort(Item* data, std::size_t n) noexcept {
    quick_sort_impl<LEAF_SIZE>(data, n);
}

void mean_pivot_quick_sort(Item* data, std::size_t n) {
    if (n <= 1) return;
    const Item initial_pivot = median_of_three(data[0], data[(n - 1) / 2], data[n - 1]);
    mean_quick_sort_impl(data, n, initial_pivot);
}

void boundary_trimmed_merge_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        insertion_sort_leaf(data, n);
        return;
    }

    AlignedBuffer buffer(n);

    // 1. 64B 캐시 라인 단위로 잘라 삽입 정렬 수행
    for (std::size_t begin = 0; begin < n; begin += LEAF_SIZE) {
        const std::size_t count = std::min<std::size_t>(LEAF_SIZE, n - begin);
        insertion_sort_leaf(data + begin, count);
    }

    // 2. 교차 포인터(left <-, right ->)로 역전된 경계만 잘라내어 양방향 병합
    for (std::size_t width = LEAF_SIZE; width < n; width *= 2) {
        for (std::size_t begin = 0; begin + width < n; begin += 2 * width) {
            const std::size_t mid = begin + width;
            const std::size_t end = std::min(n, mid + width);

            const Item left_max  = data[mid - 1];
            const Item right_min = data[mid];

            // 이미 정렬된 경계는 O(1) 비교 한 번으로 통째로 생략
            if (left_max <= right_min) continue;

            // 경계에서 바깥쪽으로 이동하며 역전이 발생한 범위만 특정 (단조 계단 함수 -> 분기 예측 극대화)
            std::size_t left_ptr  = mid;
            std::size_t right_ptr = mid;
            while (left_ptr > begin && data[left_ptr - 1] > right_min) --left_ptr;
            while (right_ptr < end  && data[right_ptr]    < left_max)  ++right_ptr;

            bidirectional_branchless_merge(data, buffer.data(), left_ptr, mid, right_ptr);
            std::memcpy(data + left_ptr, buffer.data() + left_ptr,
                        (right_ptr - left_ptr) * sizeof(Item));
        }
    }
}

void tiled_pingpong_merge_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        insertion_sort_leaf(data, n);
        return;
    }

    // 1. 거시적 역순/정렬 프로브
    if (probe_and_handle_global_trend(data, n)) return;

    AlignedBuffer buffer(n);

    // 2. L1 캐시 타일(8 KiB) 내부에서 64B 리프 정렬 및 로컬 핑퐁 병합 완료
    for (std::size_t tile_begin = 0; tile_begin < n; tile_begin += TILE_SIZE) {
        const std::size_t tile_end = std::min(n, tile_begin + TILE_SIZE);

        for (std::size_t first = tile_begin; first < tile_end; first += LEAF_SIZE) {
            const std::size_t last = std::min(tile_end, first + LEAF_SIZE);

            // 미소 블록 앞부분의 엄격한 내림차순 구간을 즉시 뒤집기
            std::size_t desc_end = first + 1;
            while (desc_end < last && data[desc_end] < data[desc_end - 1]) {
                ++desc_end;
            }
            if (desc_end - first > 1) {
                std::reverse(data + first, data + desc_end);
            }
            insertion_sort_leaf(data + first, last - first);
        }

        Item* src = data;
        Item* dst = buffer.data();
        for (std::size_t width = LEAF_SIZE; width < tile_end - tile_begin; width *= 2) {
            pingpong_merge_pass(src, dst, tile_begin, tile_end, width);
            std::swap(src, dst);
        }

        // 타일 완료 시점에 결과물이 항상 buffer 쪽에 놓이도록 패리티 맞춤
        if (src != buffer.data()) {
            std::memcpy(buffer.data() + tile_begin, src + tile_begin,
                        (tile_end - tile_begin) * sizeof(Item));
        }
    }

    // 3. 타일 간 전역 핑퐁 병합 (Global Ping-Pong Merge)
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