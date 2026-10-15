#include "benchmark_utils.hpp"
#include "hybrid_sorts.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace hsort;
using Clock = std::chrono::steady_clock;
using SortFn = void (*)(Item*, std::size_t);

struct MethodInfo {
    const char* name;
    SortFn      fn;
};

constexpr std::array<MethodInfo, 7> METHODS = {{
    {"pure_quick",           pure_quick_sort},
    {"std_sort",             std_sort_wrapper},
    {"std_stable_sort",      std_stable_sort_wrapper},
    {"cache_leaf_quick",     cache_leaf_quick_sort},
    {"mean_pivot_quick",     mean_pivot_quick_sort},
    {"trimmed_merge",        boundary_trimmed_merge_sort},
    {"tiled_pingpong_merge", tiled_pingpong_merge_sort}
}};

constexpr std::array<std::size_t, 6> BENCH_SIZES = {
    128, 2048, 8192, 131072, 1000000, 5000000
};

int main() {
    const int pinned_cpu = pin_current_thread_to_cpu(2);
    std::cout << "====================================================================\n"
              << " Benchmark Started | CPU: " << pinned_cpu
              << " | Format: " << FORMAT_NAME << " (" << sizeof(Item) << "B)\n"
              << "====================================================================\n";

    std::ofstream csv(std::string(FORMAT_NAME) + "_summary.csv");
    csv << "scenario,n,method,median_us,speedup_vs_std_sort\n";

    // 워밍업
    {
        AlignedBuffer warm(8192);
        generate_game_workload(warm.data(), 8192, 1, Scenario::Fresh3DDistances, 42);
        for (const auto& m : METHODS) m.fn(warm.data(), 8192);
    }

    constexpr int REPEATS = 5;

    for (int s = 0; s < 4; ++s) {
        const Scenario scenario = static_cast<Scenario>(s);
        std::cout << "\n>>> Scenario: " << scenario_name(scenario) << "\n"
                  << std::left << std::setw(10) << "N"
                  << std::right << std::setw(14) << "std::sort(us)"
                  << std::setw(18) << "TrimmedMerge(us)"
                  << std::setw(18) << "TiledPingPong(us)"
                  << std::setw(14) << "Best Speedup" << "\n"
                  << std::string(74, '-') << "\n";

        for (std::size_t n : BENCH_SIZES) {
            const std::size_t batch = std::max<std::size_t>(1, std::min<std::size_t>(2048, 65536 / n));
            std::array<std::vector<double>, METHODS.size()> timings;

            for (int r = 0; r < REPEATS; ++r) {
                const std::uint64_t seed = 0x900dbeefULL + n * 31 + s * 997 + r;
                AlignedBuffer source(n * batch), work(n * batch);
                generate_game_workload(source.data(), n, batch, scenario, seed);
                ReferenceData ref = build_reference(source.data(), n, batch);

                for (std::size_t m = 0; m < METHODS.size(); ++m) {
                    std::memcpy(work.data(), source.data(), n * batch * sizeof(Item));

                    asm volatile("" ::: "memory");
                    const auto start = Clock::now();
                    for (std::size_t b = 0; b < batch; ++b) {
                        METHODS[m].fn(work.data() + b * n, n);
                    }
                    asm volatile("" ::: "memory");
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

            std::array<double, METHODS.size()> medians{};
            for (std::size_t m = 0; m < METHODS.size(); ++m) {
                medians[m] = median_of(timings[m]);
            }

            const double std_sort_us = medians[1];
            for (std::size_t m = 0; m < METHODS.size(); ++m) {
                csv << scenario_name(scenario) << ',' << n << ',' << METHODS[m].name << ','
                    << std::fixed << std::setprecision(3) << medians[m] << ','
                    << (std_sort_us / medians[m]) << '\n';
            }

            const double best_custom_us = std::min(medians[5], medians[6]);
            std::cout << std::left << std::setw(10) << n
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(14) << std_sort_us
                      << std::setw(18) << medians[5]
                      << std::setw(18) << medians[6]
                      << std::setw(13) << (std_sort_us / best_custom_us) << "x\n";
        }
    }
    return 0;
}