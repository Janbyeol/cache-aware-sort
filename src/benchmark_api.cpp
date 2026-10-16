#include "hybrid_sorts.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace hsort {

// Registry of all sorting algorithms evaluated by the test and benchmark runners.
const std::vector<MethodInfo>& get_all_sort_methods() {
    static const std::vector<MethodInfo> methods = {
        {"pure_quick",           pure_quick_sort},
        {"std_sort",             std_sort_wrapper},
        {"std_stable_sort",      std_stable_sort_wrapper},
        {"cache_leaf_quick",     cache_leaf_quick_sort},
        {"mean_pivot_quick",     mean_pivot_quick_sort},
        {"trimmed_merge",        boundary_trimmed_merge_sort},
        {"tiled_pingpong_merge", tiled_pingpong_merge_sort}
    };
    return methods;
}

const char* scenario_name(Scenario s) noexcept {
    switch (s) {
        case Scenario::Fresh3DDistances: return "fresh_3d_distances";
        case Scenario::NextFrameMotion:  return "next_frame_motion";
        case Scenario::Append5PctNew:    return "append_5pct_new";
        case Scenario::DistanceBuckets:  return "distance_buckets";
        case Scenario::MostlyDescending: return "mostly_descending";
        case Scenario::GaussianNormal:   return "gaussian_normal";
        case Scenario::OrganPipe:        return "organ_pipe";
        default:                         return "unknown";
    }
}

namespace {
struct Point3D {
    std::int32_t  x, y, z, key;
    std::uint32_t id;
};

inline constexpr std::int32_t squared_distance(std::int32_t x, std::int32_t y, std::int32_t z) noexcept {
    return static_cast<std::int32_t>(
        std::int64_t{x} * x + std::int64_t{y} * y + std::int64_t{z} * z
    );
}
} // anonymous namespace

// Generates synthetic workloads modeling both game-engine spatial sorting and classic distributions.
void generate_game_workload(Item* output, std::size_t n, std::size_t batch,
                            Scenario scenario, std::uint64_t seed) {
    std::mt19937_64 rng(seed);

    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<Point3D> points(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto rand_coord = [&]() { return static_cast<std::int32_t>(rng() % 20001) - 10000; };
            points[i].x   = rand_coord();
            points[i].y   = rand_coord();
            points[i].z   = rand_coord();
            points[i].id  = static_cast<std::uint32_t>(i);
            points[i].key = squared_distance(points[i].x, points[i].y, points[i].z);
        }

        auto by_key = [](const Point3D& a, const Point3D& b) { return a.key < b.key; };

        if (scenario == Scenario::NextFrameMotion) {
            // Sort as if from the previous frame, then apply small position jitter
            std::sort(points.begin(), points.end(), by_key);
            for (auto& p : points) {
                auto jitter = [&]() { return static_cast<std::int32_t>(rng() % 11) - 5; };
                p.x  += jitter() - 3;
                p.y  += jitter() + 2;
                p.z  += jitter() - 1;
                p.key = squared_distance(p.x, p.y, p.z);
            }
        } else if (scenario == Scenario::Append5PctNew) {
            // 95% already sorted objects + 5% newly spawned unsorted objects at the end
            const std::size_t sorted_prefix = n - (n / 20);
            std::sort(points.begin(), points.begin() + sorted_prefix, by_key);
        } else if (scenario == Scenario::DistanceBuckets) {
            // Quantize distances into ~300 buckets to create heavy key duplication
            for (auto& p : points) p.key /= 1'000'000;
        } else if (scenario == Scenario::MostlyDescending) {
            // Reverse-sorted array with 1% random adjacent swaps
            std::sort(points.begin(), points.end(), [](const Point3D& a, const Point3D& b) {
                return a.key > b.key;
            });
            const std::size_t swaps = std::max<std::size_t>(1, n / 100);
            for (std::size_t s = 0; s < swaps && n >= 2; ++s) {
                const std::size_t idx = rng() % (n - 1);
                std::swap(points[idx], points[idx + 1]);
            }
        } else if (scenario == Scenario::GaussianNormal) {
            // Bell-curve distribution (-40000 to +40000) approximated via Central Limit Theorem
            for (std::size_t i = 0; i < n; ++i) {
                std::int32_t sum = 0;
                for (int k = 0; k < 4; ++k) {
                    sum += static_cast<std::int32_t>(rng() % 20001) - 10000;
                }
                points[i].key = sum;
            }
        } else if (scenario == Scenario::OrganPipe) {
            // Symmetric mountain shape: ascending first half, descending second half
            std::sort(points.begin(), points.end(), by_key);
            std::vector<Point3D> pipe(n);
            std::size_t l = 0, r = n;
            for (std::size_t i = 0; i < n; ++i) {
                if ((i & 1) == 0) pipe[l++] = points[i];
                else              pipe[--r] = points[i];
            }
            points.swap(pipe);
        }

        for (std::size_t i = 0; i < n; ++i) {
            output[b * n + i] = make_item(points[i].key, points[i].id);
        }
    }
}

// Builds ground-truth sorted keys (and payload mappings when BENCH_RECORD is enabled).
ReferenceData build_reference(const Item* source, std::size_t n, std::size_t batch) {
    ReferenceData ref;
    ref.sorted_keys.resize(n * batch);
#ifdef BENCH_RECORD
    ref.original_key_by_id.resize(n * batch);
    ref.visited_stamp.assign(n * batch, 0);
#endif

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            const Item& item = source[b * n + i];
            ref.sorted_keys[b * n + i] = get_key(item);
#ifdef BENCH_RECORD
            ref.original_key_by_id[b * n + item.entity_id] = get_key(item);
#endif
        }
        std::sort(ref.sorted_keys.begin() + b * n, ref.sorted_keys.begin() + (b + 1) * n);
    }
    return ref;
}

// Verifies that the output matches std::sort and that no 8-byte records were lost or corrupted.
void verify_sorted_batch(const Item* work, std::size_t n, std::size_t batch,
                         ReferenceData& ref, std::uint8_t stamp_id) {
    for (std::size_t b = 0; b < batch; ++b) {
        const Item* arr = work + b * n;

        if (!std::is_sorted(arr, arr + n)) {
            throw std::runtime_error("Verification failed: array is not sorted.");
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (get_key(arr[i]) != ref.sorted_keys[b * n + i]) {
                throw std::runtime_error("Verification failed: key mismatch against std::sort.");
            }
        }

#ifdef BENCH_RECORD
        for (std::size_t i = 0; i < n; ++i) {
            const auto id = arr[i].entity_id;
            if (id >= n ||
                ref.visited_stamp[b * n + id] == stamp_id ||
                get_key(arr[i]) != ref.original_key_by_id[b * n + id]) {
                throw std::runtime_error("Verification failed: entity_id corrupted or duplicated.");
            }
            ref.visited_stamp[b * n + id] = stamp_id;
        }
#else
        (void)stamp_id;
#endif
    }
}

void run_benchmark_suite() {
    using Clock = std::chrono::steady_clock;
    const auto& methods = get_all_sort_methods();
    constexpr std::array<std::size_t, 5> BENCH_SIZES = {
        128, 2048, 8192, 131072, 1000000
    };

    std::cout << "====================================================================================================\n"
              << " Benchmark Started | Format: " << FORMAT_NAME << " (" << sizeof(Item) << "B, Leaf=" << LEAF_SIZE
              << ", Tile=" << TILE_SIZE << ")\n"
              << "====================================================================================================\n";

    std::ofstream csv(std::string(FORMAT_NAME) + "_summary.csv");
    csv << "scenario,n,method,median_us,speedup_vs_std_sort\n";

    // Warm up instruction and data caches before timing
    {
        AlignedBuffer warm(8192);
        generate_game_workload(warm.data(), 8192, 1, Scenario::Fresh3DDistances, 42);
        for (const auto& m : methods) m.fn(warm.data(), 8192);
    }

    constexpr int REPEATS = 5;

    for (int s = 0; s < static_cast<int>(Scenario::Count); ++s) {
        const Scenario scenario = static_cast<Scenario>(s);
        std::cout << "\n>>> Scenario [" << s + 1 << "/" << static_cast<int>(Scenario::Count)
                  << "]: " << scenario_name(scenario) << "\n"
                  << std::left << std::setw(9) << "N"
                  << std::right
                  << std::setw(12) << "std::sort"
                  << std::setw(13) << "stable_sort"
                  << std::setw(13) << "Idea1(Leaf)"
                  << std::setw(13) << "Idea2(Mean)"
                  << std::setw(14) << "Idea3(Trim)"
                  << std::setw(14) << "Idea4(Tiled)"
                  << std::setw(13) << "Best vs std" << "\n"
                  << std::string(101, '-') << "\n";

        for (std::size_t n : BENCH_SIZES) {
            const std::size_t batch = std::max<std::size_t>(1, std::min<std::size_t>(512, 32768 / n));
            std::vector<std::vector<double>> timings(methods.size());

            for (int r = 0; r < REPEATS; ++r) {
                const std::uint64_t seed = 0x900dbeefULL + n * 31 + s * 997 + r;
                AlignedBuffer source(n * batch), work(n * batch);
                generate_game_workload(source.data(), n, batch, scenario, seed);
                ReferenceData ref = build_reference(source.data(), n, batch);

                for (std::size_t m = 0; m < methods.size(); ++m) {
                    std::memcpy(work.data(), source.data(), n * batch * sizeof(Item));

                    // Standard C++20 compiler fence to prevent dead-code elimination or reordering
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    const auto start = Clock::now();
                    for (std::size_t b = 0; b < batch; ++b) {
                        methods[m].fn(work.data() + b * n, n);
                    }
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    const auto stop = Clock::now();

                    const double per_array_us =
                        std::chrono::duration<double, std::micro>(stop - start).count() / batch;
                    verify_sorted_batch(work.data(), n, batch, ref, static_cast<std::uint8_t>(m + 1));
                    timings[m].push_back(per_array_us);
                }
            }

            auto median_of = [](std::vector<double>& v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };

            std::vector<double> medians(methods.size());
            for (std::size_t m = 0; m < methods.size(); ++m) {
                medians[m] = median_of(timings[m]);
            }

            const double std_sort_us = medians[1];
            for (std::size_t m = 0; m < methods.size(); ++m) {
                csv << scenario_name(scenario) << ',' << n << ',' << methods[m].name << ','
                    << std::fixed << std::setprecision(3) << medians[m] << ','
                    << (std_sort_us / medians[m]) << '\n';
            }

            const double best_custom_us = std::min({medians[3], medians[4], medians[5], medians[6]});
            std::cout << std::left << std::setw(9) << n
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << std_sort_us
                      << std::setw(13) << medians[2]
                      << std::setw(13) << medians[3]
                      << std::setw(13) << medians[4]
                      << std::setw(14) << medians[5]
                      << std::setw(14) << medians[6]
                      << std::setw(12) << (std_sort_us / best_custom_us) << "x\n";
        }
    }
}

} // namespace hsort
