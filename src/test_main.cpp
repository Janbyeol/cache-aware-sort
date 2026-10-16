#include "hybrid_sorts.hpp"
#include <algorithm>
#include <iostream>
#include <random>

using namespace hsort;

int main() {
    std::cout << "[Unit Test] Format: " << FORMAT_NAME
              << " (Item = " << sizeof(Item) << "B, Leaf = " << LEAF_SIZE
              << ", Tile = " << TILE_SIZE << ")\n";

    const auto& algorithms = get_all_sort_methods();
    std::mt19937_64 rng(20261015);
    std::size_t total_checks = 0;

    // 1. 경계값 크기(타일 경계, 리프 경계, 홀수/짝수) 및 4가지 게임 시나리오 전수 검사
    const std::size_t edge_sizes[] = {
        0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 511, 1023, 1024, 1025, 2047, 2048, 2049, 4097
    };

    for (std::size_t n : edge_sizes) {
        for (int s = 0; s < static_cast<int>(Scenario::Count); ++s) {
            AlignedBuffer source(n), work(n);
            generate_game_workload(source.data(), n, 1, static_cast<Scenario>(s), rng());
            ReferenceData ref = build_reference(source.data(), n, 1);

            for (std::size_t a = 0; a < algorithms.size(); ++a) {
                std::copy_n(source.data(), n, work.data());
                algorithms[a].fn(work.data(), n);
                verify_sorted_batch(work.data(), n, 1, ref, static_cast<std::uint8_t>(a + 1));
                ++total_checks;
            }
        }
    }

    // 2. 극단적 중복 키, 음수 반올림 함정 케이스, 역순 노이즈 무작위 스트레스 테스트 (300회)
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = 10 + (rng() % 3500);
        AlignedBuffer source(n), work(n);

        for (std::size_t i = 0; i < n; ++i) {
            std::int32_t key = 0;
            if (trial % 3 == 0) {
                key = static_cast<std::int32_t>(rng() % 7) - 3; // 극단적 중복 키 (-3..3)
            } else if (trial % 3 == 1) {
                key = static_cast<std::int32_t>(n - i);         // 역순 베이스
            } else {
                // 음수 나눗셈 버림 함정 케이스 (대부분 -5, 단 하나만 -4)
                key = (i == n / 2) ? -4 : -5;
            }
            source.data()[i] = make_item(key, static_cast<std::uint32_t>(i));
        }

        ReferenceData ref = build_reference(source.data(), n, 1);
        for (std::size_t a = 0; a < algorithms.size(); ++a) {
            std::copy_n(source.data(), n, work.data());
            algorithms[a].fn(work.data(), n);
            verify_sorted_batch(work.data(), n, 1, ref, static_cast<std::uint8_t>(a + 1));
            ++total_checks;
        }
    }

    std::cout << "[PASS] All " << total_checks << " verification checks succeeded!\n";
    return 0;
}