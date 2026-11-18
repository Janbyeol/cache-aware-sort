// Diagnostic for this AVX2 measurement, linked with the same Release sort objects.
#include "simd/kernels.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

using namespace hsort;

int main() {
    if (std::string_view(simd_sort_label()).find(".avx2.both.") == std::string_view::npos) {
        std::cerr << "This experiment requires the AVX2 backend with both stages enabled.\n";
        return 1;
    }
    const std::array<MethodInfo, 4> methods{{
        {"idea4", tiled_pingpong_merge_sort},
        {"leaf", [](Item* p, std::size_t n) {
            simd::tiled_sort(p, n, {simd::avx2.full_leaf, simd::scalar.merge});
        }},
        {"merge", [](Item* p, std::size_t n) {
            simd::tiled_sort(p, n, {nullptr, simd::avx2.merge});
        }},
        {"both", simd_tiled_pingpong_merge_sort}
    }};
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t n = 1'000'000;
    AlignedBuffer input(n), work(n);
    std::cout << "format,scenario,n,stage,repeat,seed,order,us_per_array\n" << std::setprecision(12);
    for (int s = 0; s < static_cast<int>(Scenario::Count); ++s) {
        const auto scenario = static_cast<Scenario>(s);
        // First pass warms every implementation on the same input.
        for (int r = -1; r < 7; ++r) {
            const std::uint64_t seed = 0x900dbeefULL + n * 31 + s * 997 + r;
            generate_game_workload(input.data(), n, 1, scenario, seed);
            auto reference = build_reference(input.data(), n, 1);
            for (std::size_t slot = 0; slot < methods.size(); ++slot) {
                const auto m = (slot + static_cast<unsigned>(r + 1)) % methods.size();
                std::copy_n(input.data(), n, work.data());
                std::atomic_signal_fence(std::memory_order_seq_cst);
                const auto start = Clock::now();
                methods[m].fn(work.data(), n);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                const auto stop = Clock::now();
                verify_sorted_batch(work.data(), n, 1, reference, static_cast<std::uint8_t>(m + 1));
                if (r >= 0) {
                    std::cout << FORMAT_NAME << ',' << scenario_name(scenario) << ',' << n << ','
                              << methods[m].name << ',' << r + 1 << ',' << seed << ',' << slot + 1
                              << ',' << std::chrono::duration<double, std::micro>(stop - start).count() << '\n';
                }
            }
        }
    }
}
