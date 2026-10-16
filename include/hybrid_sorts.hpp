#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
    #define HSORT_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
    #define HSORT_ALWAYS_INLINE __forceinline
#else
    #define HSORT_ALWAYS_INLINE inline
#endif

namespace hsort {

// ============================================================================
// 1. 정렬 대상 원소 타입 정의 (컴파일 옵션 -DBENCH_RECORD 유무로 4B / 8B 전환)
// ============================================================================
#ifdef BENCH_RECORD
struct alignas(8) DistanceRecord {
    std::int32_t  key;
    std::uint32_t entity_id;

    DistanceRecord() = default;
    constexpr DistanceRecord(std::int32_t k, std::uint32_t id = 0) noexcept
        : key(k), entity_id(id) {}

    constexpr bool operator<(const DistanceRecord& other) const noexcept  { return key < other.key; }
    constexpr bool operator>(const DistanceRecord& other) const noexcept  { return key > other.key; }
    constexpr bool operator<=(const DistanceRecord& other) const noexcept { return key <= other.key; }
    constexpr bool operator>=(const DistanceRecord& other) const noexcept { return key >= other.key; }
    constexpr bool operator==(const DistanceRecord& other) const noexcept { return key == other.key; }
};

using Item = DistanceRecord;
constexpr const char* FORMAT_NAME = "key_id_8b";
#else
using Item = std::int32_t;
constexpr const char* FORMAT_NAME = "key_only_4b";
#endif

static_assert(std::is_trivially_copyable_v<Item>, "Item must be trivially copyable.");

inline constexpr std::int32_t get_key(const Item& item) noexcept {
#ifdef BENCH_RECORD
    return item.key;
#else
    return item;
#endif
}

inline constexpr Item make_item(std::int32_t key, std::uint32_t id = 0) noexcept {
#ifdef BENCH_RECORD
    return DistanceRecord{key, id};
#else
    (void)id;
    return key;
#endif
}

// ============================================================================
// 2. 하드웨어 캐시 계층 상수 (64B Cache Line & 8KiB L1 Tile)
// ============================================================================
constexpr std::size_t CACHE_LINE_BYTES = 64;
constexpr std::size_t LEAF_SIZE        = CACHE_LINE_BYTES / sizeof(Item); // 4B -> 16개, 8B -> 8개
constexpr std::size_t L1_TILE_BYTES    = 8192;                            // 8 KiB
constexpr std::size_t TILE_SIZE        = L1_TILE_BYTES / sizeof(Item);    // 4B -> 2048개, 8B -> 1024개

// ============================================================================
// 3. 64바이트 캐시 라인 경계 정렬 메모리 버퍼 (RAII)
// ============================================================================
class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t count)
        : size_(count) {
        const std::size_t alloc_count = (count > 0) ? count : 1;
        data_ = static_cast<Item*>(
            ::operator new[](alloc_count * sizeof(Item), std::align_val_t{CACHE_LINE_BYTES})
        );
    }

    ~AlignedBuffer() {
        ::operator delete[](data_, std::align_val_t{CACHE_LINE_BYTES});
    }

    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    Item*       data() noexcept       { return data_; }
    const Item* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    Item*       data_ = nullptr;
    std::size_t size_ = 0;
};

// ============================================================================
// 4. 공통 인라인 커널 (각 정렬 소스 파일에서 공유)
// ============================================================================
namespace detail {

HSORT_ALWAYS_INLINE Item median_of_three(Item a, Item b, Item c) noexcept {
    if (a < b) return (b < c) ? b : std::max(a, c);
    return (a < c) ? a : std::max(b, c);
}

// 64B 캐시 라인(16개 또는 8개 원소) 내부를 처리하는 고속 선형 삽입 정렬
HSORT_ALWAYS_INLINE void insertion_sort_leaf(Item* __restrict data, std::size_t n) noexcept {
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
HSORT_ALWAYS_INLINE std::size_t hoare_partition(Item* __restrict data, std::size_t n, Item pivot) noexcept {
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
inline void quick_sort_impl(Item* data, std::size_t n) noexcept {
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

// 양방향 분기 제거 병합 (Bidirectional Branchless Merge)
// 전면 포인터(최솟값 ->)와 후면 포인터(최댓값 <-)를 동시에 이동시켜 ILP 2배 확보
inline void bidirectional_branchless_merge(const Item* __restrict src, Item* __restrict dst,
                                           std::size_t begin, std::size_t mid, std::size_t end) noexcept {
    std::size_t l_fwd   = begin, l_bwd   = mid;
    std::size_t r_fwd   = mid,   r_bwd   = end;
    std::size_t out_fwd = begin, out_bwd = end;

    while (l_fwd < l_bwd && r_fwd < r_bwd) {
        const std::size_t safe_pairs = std::min(l_bwd - l_fwd, r_bwd - r_fwd);

        for (std::size_t s = 0; s < safe_pairs; ++s) {
            // 1. 앞쪽에서 최솟값 선택 (cmov 유도, 동일 키일 때 좌측 우선 선택)
            const Item lv         = src[l_fwd];
            const Item rv         = src[r_fwd];
            const bool take_right = (rv < lv);
            dst[out_fwd++]        = take_right ? rv : lv;
            l_fwd                += !take_right;
            r_fwd                += take_right;

            // 2. 뒤쪽에서 최댓값 선택 (동일 키일 때 우측 우선 선택하여 안정성 보장)
            const Item lb        = src[l_bwd - 1];
            const Item rb        = src[r_bwd - 1];
            const bool take_left = (lb > rb);
            dst[--out_bwd]       = take_left ? lb : rb;
            l_bwd               -= take_left;
            r_bwd               -= !take_left;
        }
    }

    if (l_fwd < l_bwd) {
        std::memcpy(dst + out_fwd, src + l_fwd, (l_bwd - l_fwd) * sizeof(Item));
    } else if (r_fwd < r_bwd) {
        std::memcpy(dst + out_fwd, src + r_fwd, (r_bwd - r_fwd) * sizeof(Item));
    }
}

} // namespace detail

// ============================================================================
// 5. 정렬 알고리즘 공개 API (각 아이디어별 개별 소스 파일에 구현)
// ============================================================================

// [Baselines] 비교군 정렬 (src/baseline_sorts.cpp)
void pure_quick_sort(Item* data, std::size_t n) noexcept;
void std_sort_wrapper(Item* data, std::size_t n) noexcept;
void std_stable_sort_wrapper(Item* data, std::size_t n);

// [Idea 1] 64B 캐시 라인 리프 제한 + 선형 삽입 하이브리드 퀵 정렬 (src/idea1_cache_leaf_quick.cpp)
void cache_leaf_quick_sort(Item* data, std::size_t n) noexcept;

// [Idea 2] 무임승차 합산(Summation) 평균 피벗 예측 퀵 정렬 (src/idea2_mean_pivot_quick.cpp)
void mean_pivot_quick_sort(Item* data, std::size_t n) noexcept;

// [Idea 3] 64B 블록 정렬 + 역방향 교차 포인터 경계 트리밍 병합 정렬 (src/idea3_boundary_trimmed_merge.cpp)
void boundary_trimmed_merge_sort(Item* data, std::size_t n);

// [Idea 4] 거시적 경향 감지 + L1 타일링 + 이진 트리밍 + 핑퐁 병합 정렬 (src/idea4_tiled_pingpong_merge.cpp)
void tiled_pingpong_merge_sort(Item* data, std::size_t n);

// ============================================================================
// 6. 벤치마크 및 검증 유틸리티 API (src/benchmark_api.cpp)
// ============================================================================
using SortFn = void (*)(Item*, std::size_t);

struct MethodInfo {
    const char* name;
    SortFn      fn;
};

// 벤치마크 및 테스트에서 공통으로 순회할 정렬 함수 목록 반환
const std::vector<MethodInfo>& get_all_sort_methods();

enum class Scenario : int {
    Fresh3DDistances = 0, // 완전 무작위 3D 거리제곱 (Uniform 3D coords)
    NextFrameMotion  = 1, // 이전 프레임 정렬 후 ±5cm 미세 이동 (Nearly Sorted)
    Append5PctNew    = 2, // 앞쪽 95% 정렬 + 뒤쪽 5% 신규 스폰 (Sorted Prefix + Random Tail)
    DistanceBuckets  = 3, // 0..300 구간 다중 중복 키 (Many Duplicates)
    MostlyDescending = 4, // 역순 정렬 베이스 + 1% 미세 노이즈 (Reverse / Descending)
    GaussianNormal   = 5, // 평균 중심 정규분포 (Bell-curve / Normal Distribution)
    OrganPipe        = 6, // 오르간 파이프 분포 (오름차순 후 내림차순 산 모양)
    Count            = 7
};

const char* scenario_name(Scenario s) noexcept;

void generate_game_workload(Item* output, std::size_t n, std::size_t batch,
                            Scenario scenario, std::uint64_t seed);

struct ReferenceData {
    std::vector<std::int32_t> sorted_keys;
#ifdef BENCH_RECORD
    std::vector<std::int32_t> original_key_by_id;
    std::vector<std::uint8_t> visited_stamp;
#endif
};

ReferenceData build_reference(const Item* source, std::size_t n, std::size_t batch);

void verify_sorted_batch(const Item* work, std::size_t n, std::size_t batch,
                         ReferenceData& ref, std::uint8_t stamp_id);

// 전체 벤치마크 스위트 실행 및 CSV 저장 API
void run_benchmark_suite();

} // namespace hsort