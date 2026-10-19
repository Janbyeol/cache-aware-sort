#include "hybrid_sorts.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace hsort;

namespace {

const char* detect_arch_label() noexcept {
#if defined(_M_ARM64) || defined(__aarch64__)
    return "ARM64_Oryon";
#elif defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#else
    return "Unknown_Arch";
#endif
}

const char* scenario_slug(Scenario s) noexcept {
    switch (s) {
        case Scenario::Fresh3DDistances: return "0_Uniform_Random";
        case Scenario::NextFrameMotion:  return "1_Frame_Jitter_Nearly_Sorted";
        case Scenario::Append5PctNew:    return "2_Append_5Pct_New_Tail";
        case Scenario::DistanceBuckets:  return "3_Distance_Buckets_Duplicates";
        case Scenario::MostlyDescending: return "4_Mostly_Descending";
        case Scenario::GaussianNormal:   return "5_Gaussian_Normal";
        case Scenario::OrganPipe:        return "6_Organ_Pipe_Symmetric";
        default:                         return "Unknown";
    }
}

#ifdef _WIN32
void pin_thread_for_snapdragon() noexcept {
    // Lock benchmark to a single CPU core so Windows 11 does not migrate the thread
    // across Snapdragon X Elite's 4-core Oryon L2 clusters during measurement.
    ::SetPriorityClass(::GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    ::SetThreadAffinityMask(::GetCurrentThread(), static_cast<DWORD_PTR>(1ULL));
}

inline std::uint64_t read_thread_cycles() noexcept {
    ULONG64 cycles = 0;
    ::QueryThreadCycleTime(::GetCurrentThread(), &cycles);
    return static_cast<std::uint64_t>(cycles);
}

inline std::uint64_t read_qpc_ticks() noexcept {
    LARGE_INTEGER li{};
    ::QueryPerformanceCounter(&li);
    return static_cast<std::uint64_t>(li.QuadPart);
}

inline double qpc_to_microseconds(std::uint64_t ticks, std::uint64_t freq) noexcept {
    return (static_cast<double>(ticks) * 1'000'000.0) / static_cast<double>(freq);
}
#endif

} // anonymous namespace

int main(int argc, char** argv) {
#ifndef _WIN32
    std::cerr << "This benchmark binary is designed for Windows (ARM64 / x64).\n";
    return 1;
#else
    pin_thread_for_snapdragon();

    const std::size_t n       = (argc >= 2) ? std::stoull(argv[1]) : 1000000;
    const int         repeats = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 3;
    const std::string csv_out = (argc >= 4) ? argv[3]              : "pure_cache_results.csv";
    const std::string md_out  = (argc >= 5) ? argv[4]              : "pure_cache_results.md";
    const bool        append  = (argc >= 6 && std::string(argv[5]) == "--append");

#ifdef BENCH_RECORD
    const char* dtype_label = "8B_Record";
#else
    const char* dtype_label = "4B_Int32";
#endif
    const char* arch_label  = detect_arch_label();

    LARGE_INTEGER freq_li{};
    ::QueryPerformanceFrequency(&freq_li);
    const std::uint64_t qpc_freq = static_cast<std::uint64_t>(freq_li.QuadPart);

    struct AlgoEntry {
        const char* label;
        SortFn      fn;
    };
    const std::array<AlgoEntry, 6> algos = {{
        {"std::sort",              std_sort_wrapper},
        {"std::stable_sort",       std_stable_sort_wrapper},
        {"cache_leaf_quick",       cache_leaf_quick_sort},
        {"mean_pivot_quick",       mean_pivot_quick_sort},
        {"boundary_trimmed_merge", boundary_trimmed_merge_sort},
        {"tiled_pingpong_merge",   tiled_pingpong_merge_sort}
    }};

    std::ofstream csv(csv_out, append ? std::ios::app : std::ios::trunc);
    std::ofstream md(md_out,   append ? std::ios::app : std::ios::trunc);

    if (!append) {
        csv << "Data_Type,Arch,Scenario,N,Algorithm,Runtime_us,Runtime_ms,Speedup_Time,Cycles,Speedup_Cycles,Cycles_Per_Elem,Ns_Per_Elem,Throughput_Mops\n";

        md << "# Windows ARM64 (Snapdragon X Elite) Pure Sorting Analysis (N = " << n << ", Repeats = " << repeats << ")\n\n"
           << "> `generate_game_workload()` 및 배열 복사(`memcpy`) 구간을 100% 제외한 **순수 정렬 함수 실행 구간의 하드웨어 Thread Cycles(`QueryThreadCycleTime`) 및 고정밀 런타임(`QPC`) 측정치**입니다.\n\n"
           << "| Data Type | Arch | Scenario | Algorithm | Runtime (us) | Speedup (Time) | Cycles | Speedup (Cycles) | Cycles/Elem | Throughput (M/s) |\n"
           << "| :--- | :--- | :--- | :--- | ---: | :---: | ---: | :---: | ---: | ---: |\n";
    }

    // Warm up CPU frequency and instruction cache before starting measurements
    {
        constexpr std::size_t warm_n = 8192;
        AlignedBuffer warm(warm_n);
        generate_game_workload(warm.data(), warm_n, 1, Scenario::Fresh3DDistances, 42);
        for (const auto& a : algos) {
            a.fn(warm.data(), warm_n);
        }
    }

    // For tiny N, batch multiple pre-copied arrays so timer/cycle quantization is eliminated
    // while keeping 100% of memcpy operations outside the measured region.
    const std::size_t batch = (n < 32768)
        ? std::max<std::size_t>(1, std::min<std::size_t>(512, 32768 / std::max<std::size_t>(1, n)))
        : 1;

    AlignedBuffer master(n * batch);
    AlignedBuffer work(n * batch);

    for (int s = 0; s < static_cast<int>(Scenario::Count); ++s) {
        const Scenario scenario = static_cast<Scenario>(s);

        // Generate workload ONCE per scenario outside the measurement window
        generate_game_workload(master.data(), n, batch, scenario, 42 + static_cast<std::uint64_t>(s) * 997);

        double std_sort_us     = 0.0;
        double std_sort_cycles = 0.0;

        for (std::size_t a_idx = 0; a_idx < algos.size(); ++a_idx) {
            const auto& algo = algos[a_idx];
            std::vector<double> sample_us;
            std::vector<double> sample_cycles;
            sample_us.reserve(static_cast<std::size_t>(repeats));
            sample_cycles.reserve(static_cast<std::size_t>(repeats));

            for (int r = 0; r < repeats; ++r) {
                // Copy unsorted master buffer into work buffer BEFORE starting counters
                std::memcpy(work.data(), master.data(), n * batch * sizeof(Item));

                std::atomic_signal_fence(std::memory_order_seq_cst);
                const std::uint64_t qpc_start = read_qpc_ticks();
                const std::uint64_t cyc_start = read_thread_cycles();
                std::atomic_signal_fence(std::memory_order_seq_cst);

                for (std::size_t b = 0; b < batch; ++b) {
                    algo.fn(work.data() + b * n, n);
                }

                std::atomic_signal_fence(std::memory_order_seq_cst);
                const std::uint64_t cyc_end = read_thread_cycles();
                const std::uint64_t qpc_end = read_qpc_ticks();
                std::atomic_signal_fence(std::memory_order_seq_cst);

                const double elapsed_us = qpc_to_microseconds(qpc_end - qpc_start, qpc_freq) / static_cast<double>(batch);
                const double elapsed_cyc = static_cast<double>(cyc_end - cyc_start) / static_cast<double>(batch);

                sample_us.push_back(elapsed_us);
                sample_cycles.push_back(elapsed_cyc);
            }

            // Use median across repeats to suppress OS interrupt spikes
            std::sort(sample_us.begin(), sample_us.end());
            std::sort(sample_cycles.begin(), sample_cycles.end());

            const double med_us     = sample_us[sample_us.size() / 2];
            const double med_cycles = sample_cycles[sample_cycles.size() / 2];

            if (a_idx == 0) {
                std_sort_us     = med_us;
                std_sort_cycles = med_cycles;
            }

            const double runtime_ms     = med_us / 1000.0;
            const double speedup_time   = (med_us > 0.0)     ? (std_sort_us / med_us)         : 0.0;
            const double speedup_cycles = (med_cycles > 0.0) ? (std_sort_cycles / med_cycles) : 0.0;
            const double cyc_per_elem   = (n > 0)            ? (med_cycles / static_cast<double>(n)) : 0.0;
            const double ns_per_elem    = (n > 0)            ? ((med_us * 1000.0) / static_cast<double>(n)) : 0.0;
            const double mops           = (med_us > 0.0)     ? (static_cast<double>(n) / med_us) : 0.0;

            char csv_line[512];
            std::snprintf(
                csv_line, sizeof(csv_line),
                "%s,%s,%s,%zu,%s,%.3f,%.4f,%.2fx,%.0f,%.2fx,%.2f,%.2f,%.2f\n",
                dtype_label, arch_label, scenario_slug(scenario), n, algo.label,
                med_us, runtime_ms, speedup_time,
                med_cycles, speedup_cycles,
                cyc_per_elem, ns_per_elem, mops
            );
            csv << csv_line;

            char md_line[512];
            std::snprintf(
                md_line, sizeof(md_line),
                "| %s | %s | %s | `%s` | %.3f | **%.2fx** | %.0f | **%.2fx** | %.2f | %.2f |\n",
                dtype_label, arch_label, scenario_slug(scenario), algo.label,
                med_us, speedup_time, med_cycles, speedup_cycles, cyc_per_elem, mops
            );
            md << md_line;

            std::printf(
                "%-9s | %-29s | %-22s | %10.2f us | %8.2fx | %11.0f | %8.2fx | %8.2f M/s\n",
                dtype_label, scenario_slug(scenario), algo.label,
                med_us, speedup_time, med_cycles, speedup_cycles, mops
            );
        }
    }

    return 0;
#endif
}
