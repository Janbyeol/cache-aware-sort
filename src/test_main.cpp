#include "benchmark_utils.hpp"
#include "hybrid_sorts.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <random>

using namespace hsort;

using SortFn = void (*)(Item*, std::size_t);

struct AlgorithmEntry {
    const char* name;
    SortFn      fn;
};

constexpr std::array<AlgorithmEntry, 7> ALGORITHMS = {{
    {"Pure QuickSort",          pure_quick_sort},
    {"std::sort",               std_sort_wrapper},
    {"std::stable_sort",        std_stable_sort_wrapper},
    {"64B Leaf QuickSort",      cache_leaf_quick_sort},
    {"Mean-Pivot QuickSort",    mean_pivot_quick_sort},
    {"Boundary-Trimmed Merge",  boundary_trimmed_merge_sort},
    {"Tiled Ping-Pong Merge",   tiled_pingpong_merge_sort}
}};

int main() {
    std::cout << "[Unit Test] Format: " << FORMAT_NAME
              << " (Item = " << sizeof(Item) << "B, Leaf = " << LEAF_SIZE
              << ", Tile = " << TILE_SIZE << ")\n";

    std::mt19937_64 rng(20261015);
    std::size_t total_checks = 0;

    // 1. 경계값 크기(타일 경계, 리프 경계, 홀수/짝수) 및 4가지 게임 시나리오 전수 검사
    const std::size_t edge_sizes[] = {
        0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 511, 1023, 1024, 1025, 2047, 2048, 2049, 4097
    };

    for (std::size_t n : edge_sizes) {
        for (int s = 0; s < 4; ++s) {
            AlignedBuffer source(n), work(n);
            generate_game_workload(source.data(), n, 1, static_cast<Scenario>(s), rng());
            ReferenceData ref = build_reference(source.data(), n, 1);

            for (std::size_t a = 0; a < ALGORITHMS.size(); ++a) {
                std::copy_n(source.data(), n, work.data());
                ALGORITHMS[a].fn(work.data(), n);
                verify_sorted_batch(work.data(), n, 1, ref, static_cast<std::uint8_t>(a + 1));
                ++total_checks;
            }
        }
    }

    // 2. 극단적 중복 키 및 역순 노이즈 무작위 스트레스 테스트 (300회)
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = 10 + (rng() % 3500);
        AlignedBuffer source(n), work(n);

        for (std::size_t i = 0; i < n; ++i) {
            const std::int32_t key = (trial % 2 == 0)
                ? static_cast<std::int32_t>(rng() % 7) - 3 // 극단적 중복 키 (-3..3)
                : static_cast<std::int32_t>(n - i);        // 역순 베이스
            source.data()[i] = make_item(key, static_cast<std::uint32_t>(i));
        }

        ReferenceData ref = build_reference(source.data(), n, 1);
        for (std::size_t a = 0; a < ALGORITHMS.size(); ++a) {
            std::copy_n(source.data(), n, work.data());
            ALGORITHMS[a].fn(work.data(), n);
            verify_sorted_batch(work.data(), n, 1, ref, static_cast<std::uint8_t>(a + 1));
            ++total_checks;
        }
    }

    std::cout << "[PASS] All " << total_checks << " verification checks succeeded!\n";
    return 0;
}