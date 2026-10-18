#include "hybrid_sorts.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace hsort;

namespace {

std::size_t verify_trimmed_merge_regressions() {
    std::mt19937_64 rng(0x3b0a17dULL);
    const std::size_t sizes[] = {
        0, 1, LEAF_SIZE - 1, LEAF_SIZE, LEAF_SIZE + 1,
        2 * LEAF_SIZE - 1, 2 * LEAF_SIZE, 2 * LEAF_SIZE + 1,
        127, 128, 129, TILE_SIZE - 1, TILE_SIZE, TILE_SIZE + 1,
        2 * TILE_SIZE - 1, 2 * TILE_SIZE, 2 * TILE_SIZE + 1, 4 * TILE_SIZE + 1
    };
    std::size_t checks = 0;
    for (std::size_t n : sizes) {
        for (int pattern = 0; pattern < 8; ++pattern) {
            // Offset by one item to cover inputs that are not cache-line aligned.
            AlignedBuffer storage(n + 2);
            const Item guard = make_item(std::numeric_limits<std::int32_t>::max(),
                                         std::numeric_limits<std::uint32_t>::max());
            storage.data()[0] = guard;
            storage.data()[n + 1] = guard;
            Item* const data = storage.data() + 1;
            for (std::size_t i = 0; i < n; ++i) {
                std::int32_t key = 0;
                switch (pattern) {
                    case 0: key = -7; break;
                    case 1:
                        key = (i & 1) ? std::numeric_limits<std::int32_t>::min()
                                      : std::numeric_limits<std::int32_t>::max();
                        break;
                    case 2: key = static_cast<std::int32_t>(n - i); break;
                    case 3: key = static_cast<std::int32_t>(i < n / 2 ? 2 * i : 2 * (i - n / 2) + 1); break;
                    case 4: key = static_cast<std::int32_t>(i < n / 2 ? (i < n / 4 ? 0 : n) : n / 2); break;
                    case 5: key = static_cast<std::int32_t>(i < n - n / 20 ? i : rng() % (n + 1)); break;
                    case 6: key = static_cast<std::int32_t>(rng() % 11) - 5; break;
                    case 7: key = static_cast<std::int32_t>(rng() % 2'000'001) - 1'000'000; break;
                }
                data[i] = make_item(key, static_cast<std::uint32_t>(i));
            }
            std::vector<Item> expected(data, data + n);
            std::stable_sort(expected.begin(), expected.end());
            boundary_trimmed_merge_sort(data, n);
            for (std::size_t i = 0; i < n; ++i) {
                if (get_key(data[i]) != get_key(expected[i]))
                    throw std::runtime_error("Trimmed merge regression: incorrect key.");
#ifdef BENCH_RECORD
                if (data[i].entity_id != expected[i].entity_id)
                    throw std::runtime_error("Trimmed merge regression: record identity or tie order changed.");
#endif
            }
            if (get_key(storage.data()[0]) != get_key(guard) ||
                get_key(storage.data()[n + 1]) != get_key(guard))
                throw std::runtime_error("Trimmed merge regression: guard key overwritten.");
#ifdef BENCH_RECORD
            if (storage.data()[0].entity_id != guard.entity_id ||
                storage.data()[n + 1].entity_id != guard.entity_id)
                throw std::runtime_error("Trimmed merge regression: guard payload overwritten.");
#endif
            ++checks;
        }
    }
    return checks;
}

void verify_allocation_overflow_is_rejected() {
    constexpr std::size_t excessive_count =
        std::numeric_limits<std::size_t>::max() / sizeof(Item) + 1;
    bool rejected = false;
    try {
        AlignedBuffer buffer(excessive_count);
    } catch (const std::bad_array_new_length&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("AlignedBuffer accepted a byte-count overflow.");
}

} // namespace

int main() {
    std::cout << "[Unit Test] Format: " << FORMAT_NAME
              << " (Item = " << sizeof(Item) << "B, Leaf = " << LEAF_SIZE
              << ", Tile = " << TILE_SIZE << ")\n";

    const auto& algorithms = get_all_sort_methods();
    std::mt19937_64 rng(20261015);
    std::size_t total_checks = verify_trimmed_merge_regressions();

    // 1. Exhaustive boundary size tests (0, 1, odd/even, cache-line & L1-tile boundaries)
    //    across all 7 input distribution scenarios.
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

    // 2. Randomized stress tests (300 trials) covering:
    //    - Extreme duplicate keys (-3..3)
    //    - Strictly descending arrays
    //    - Negative integer division truncation traps (nineteen -5s and one -4)
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = 10 + (rng() % 3500);
        AlignedBuffer source(n), work(n);

        for (std::size_t i = 0; i < n; ++i) {
            std::int32_t key = 0;
            if (trial % 3 == 0) {
                key = static_cast<std::int32_t>(rng() % 7) - 3;
            } else if (trial % 3 == 1) {
                key = static_cast<std::int32_t>(n - i);
            } else {
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

    verify_allocation_overflow_is_rejected();
    ++total_checks;
    std::cout << "[PASS] All " << total_checks << " verification checks succeeded!\n";
    return 0;
}
