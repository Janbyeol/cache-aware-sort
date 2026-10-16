#include "hybrid_sorts.hpp"

namespace hsort {

namespace {

// ============================================================================
// [Idea 4] 거시적 경향 감지 + L1 타일링 + 경계 트리밍 + 핑퐁 병합 정렬
//   - Idea 3에서 배열이 커질 때 발생하는 두 가지 병목(캐시 방출, 매 단계 memcpy)을 해결:
//     1. 8KiB L1 캐시 타일(TILE_SIZE) 단위로 묶어 타일 내부에서 64B 정렬과 병합을 먼저 끝냄
//     2. 원본 배열(data)과 보조 버퍼(buffer)를 번갈아 쓰는 Ping-Pong Merge로 중간 복사 제거
//     3. 블록 크기가 커졌을 때 경계 트리밍을 이진 탐색(upper_bound / lower_bound)으로 가속
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
    const std::size_t trim_left = static_cast<std::size_t>(
        std::upper_bound(src + begin, src + mid, src[mid]) - src
    );
    const std::size_t trim_right = static_cast<std::size_t>(
        std::lower_bound(src + mid, src + end, src[mid - 1]) - src
    );

    if (trim_left > begin) {
        std::memcpy(dst + begin, src + begin, (trim_left - begin) * sizeof(Item));
    }
    if (trim_right < end) {
        std::memcpy(dst + trim_right, src + trim_right, (end - trim_right) * sizeof(Item));
    }

    detail::bidirectional_branchless_merge(src, dst, trim_left, mid, trim_right);
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

void tiled_pingpong_merge_sort(Item* data, std::size_t n) {
    if (n <= LEAF_SIZE) {
        detail::insertion_sort_leaf(data, n);
        return;
    }

    // 1단계: 거시적 역순/정렬 프로브
    if (probe_and_handle_global_trend(data, n)) return;

    AlignedBuffer buffer(n);

    // 2단계: L1 캐시 타일(8 KiB) 내부에서 64B 리프 정렬 및 로컬 핑퐁 병합 완료
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
            detail::insertion_sort_leaf(data + first, last - first);
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

    // 3단계: 타일 간 전역 핑퐁 병합 (Global Ping-Pong Merge)
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
