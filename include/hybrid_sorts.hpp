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
// 1. Element Type Definition (4B int32_t vs. 8B Key+ID Record)
//    Compile with -DBENCH_RECORD=1 to test 8-byte key-value payloads.
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
// 2. Hardware Cache Hierarchy Constants
//    - CACHE_LINE_BYTES (64B): Standard CPU cache line size (16x 4B or 8x 8B)
//    - L1_TILE_BYTES (8KiB):   Working set size that comfortably fits in L1D
//                              alongside its auxiliary ping-pong merge buffer.
// ============================================================================
constexpr std::size_t CACHE_LINE_BYTES = 64;
constexpr std::size_t LEAF_SIZE        = CACHE_LINE_BYTES / sizeof(Item); // 16 items (4B) or 8 items (8B)
constexpr std::size_t L1_TILE_BYTES    = 8192;                            // 8 KiB
constexpr std::size_t TILE_SIZE        = L1_TILE_BYTES / sizeof(Item);    // 2048 items (4B) or 1024 items (8B)

// ============================================================================
// 3. 64-Byte Cache-Line Aligned Scratch Buffer (RAII)
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
// 4. Shared Low-Level Kernels
// ============================================================================
namespace detail {

HSORT_ALWAYS_INLINE Item median_of_three(Item a, Item b, Item c) noexcept {
    if (a < b) return (b < c) ? b : std::max(a, c);
    return (a < c) ? a : std::max(b, c);
}

// Fast linear insertion sort for micro-blocks that already reside in a single 64B cache line.
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

// Classic Hoare partition scheme.
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

// Bidirectional Branchless Merge:
// Merges [begin, mid) and [mid, end) from `src` into `dst`.
// By picking the minimum from the front (->) and the maximum from the back (<-) simultaneously
// using branchless conditional moves (cmov), we break serial data dependencies and double ILP.
inline void bidirectional_branchless_merge(const Item* __restrict src, Item* __restrict dst,
                                           std::size_t begin, std::size_t mid, std::size_t end) noexcept {
    std::size_t l_fwd   = begin, l_bwd   = mid;
    std::size_t r_fwd   = mid,   r_bwd   = end;
    std::size_t out_fwd = begin, out_bwd = end;

    while (l_fwd < l_bwd && r_fwd < r_bwd) {
        // Number of pairs we can process without checking for range exhaustion mid-loop
        const std::size_t safe_pairs = std::min(l_bwd - l_fwd, r_bwd - r_fwd);

        for (std::size_t s = 0; s < safe_pairs; ++s) {
            // 1. Front pointer: pick the smaller element (advances left-to-right ->)
            const Item lv         = src[l_fwd];
            const Item rv         = src[r_fwd];
            const bool take_right = (rv < lv);
            dst[out_fwd++]        = take_right ? rv : lv;
            l_fwd                += !take_right;
            r_fwd                += take_right;

            // 2. Back pointer: pick the larger element (advances right-to-left <-)
            //    Completely independent of the front pointer, enabling parallel execution per cycle.
            const Item lb        = src[l_bwd - 1];
            const Item rb        = src[r_bwd - 1];
            const bool take_left = (lb > rb);
            dst[--out_bwd]       = take_left ? lb : rb;
            l_bwd               -= take_left;
            r_bwd               -= !take_left;
        }
    }

    // Copy any remaining unmerged elements in the middle
    if (l_fwd < l_bwd) {
        std::memcpy(dst + out_fwd, src + l_fwd, (l_bwd - l_fwd) * sizeof(Item));
    } else if (r_fwd < r_bwd) {
        std::memcpy(dst + out_fwd, src + r_fwd, (r_bwd - r_fwd) * sizeof(Item));
    }
}

} // namespace detail

// ============================================================================
// 5. Public Sorting APIs (Implemented in separate files per idea)
// ============================================================================

// [Baselines] Reference sorting algorithms (src/baseline_sorts.cpp)
void pure_quick_sort(Item* data, std::size_t n) noexcept;
void std_sort_wrapper(Item* data, std::size_t n) noexcept;
void std_stable_sort_wrapper(Item* data, std::size_t n);

// [Idea 1] 64B Cache-Line Cutoff + Linear Insertion Hybrid Quick Sort (src/idea1_cache_leaf_quick.cpp)
void cache_leaf_quick_sort(Item* data, std::size_t n) noexcept;

// [Idea 2] Piggybacked Summation Mean-Pivot Quick Sort (src/idea2_mean_pivot_quick.cpp)
void mean_pivot_quick_sort(Item* data, std::size_t n) noexcept;

// [Idea 3] 64B Micro-Block Sort + Cross-Pointer Boundary-Trimmed Merge Sort (src/idea3_boundary_trimmed_merge.cpp)
void boundary_trimmed_merge_sort(Item* data, std::size_t n);

// [Idea 4] Global Trend Probe + 8KiB L1 Tiling + Ping-Pong Merge Sort (src/idea4_tiled_pingpong_merge.cpp)
void tiled_pingpong_merge_sort(Item* data, std::size_t n);

// ============================================================================
// 6. Benchmark & Verification API (src/benchmark_api.cpp)
// ============================================================================
using SortFn = void (*)(Item*, std::size_t);

struct MethodInfo {
    const char* name;
    SortFn      fn;
};

const std::vector<MethodInfo>& get_all_sort_methods();

enum class Scenario : int {
    Fresh3DDistances = 0, // Uniformly random 3D squared distances
    NextFrameMotion  = 1, // Pre-sorted from previous frame + small position jitter (Nearly Sorted)
    Append5PctNew    = 2, // 95% sorted prefix + 5% newly spawned random tail
    DistanceBuckets  = 3, // Quantized distance buckets (High duplicate keys)
    MostlyDescending = 4, // Reverse-sorted array with 1% random swap noise
    GaussianNormal   = 5, // Bell-curve normal distribution (CLT sum)
    OrganPipe        = 6, // Symmetric mountain distribution (Ascending first half, Descending second half)
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

void run_benchmark_suite();

} // namespace hsort