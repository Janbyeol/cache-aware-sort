#include "hybrid_sorts.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace hsort;

namespace {

#ifdef __linux__
long perf_event_open(struct perf_event_attr* hw_event, pid_t pid,
                     int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int open_pmu_counter(std::uint32_t type, std::uint64_t config) {
    struct perf_event_attr pe{};
    pe.type           = type;
    pe.size           = sizeof(struct perf_event_attr);
    pe.config         = config;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    pe.read_format    = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(perf_event_open(&pe, 0, -1, -1, 0));
}

constexpr std::uint64_t make_cache_config(std::uint64_t cache_id,
                                          std::uint64_t op_id,
                                          std::uint64_t result_id) {
    return cache_id | (op_id << 8) | (result_id << 16);
}

struct ReadFormat {
    std::uint64_t value        = 0;
    std::uint64_t time_enabled = 0;
    std::uint64_t time_running = 0;
};

double read_pmu_counter(int fd) {
    if (fd < 0) return 0.0;
    ReadFormat rf{};
    if (::read(fd, &rf, sizeof(rf)) != sizeof(rf)) return 0.0;
    if (rf.time_running == 0) return 0.0;
    if (rf.time_running < rf.time_enabled) {
        return static_cast<double>(rf.value) *
               (static_cast<double>(rf.time_enabled) / static_cast<double>(rf.time_running));
    }
    return static_cast<double>(rf.value);
}
#endif

struct PurePerfMetrics {
    double cycles        = 0.0;
    double instructions  = 0.0;
    double l1_loads      = 0.0;
    double l1_misses     = 0.0;
    double llc_refs      = 0.0;
    double llc_misses    = 0.0;
    double branches      = 0.0;
    double branch_misses = 0.0;
};

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

} // anonymous namespace

int main(int argc, char** argv) {
#ifndef __linux__
    std::cerr << "This tool requires Linux (perf_event_open).\n";
    return 1;
#else
    const std::size_t n       = (argc >= 2) ? std::stoull(argv[1]) : 1000000;
    const int         repeats = (argc >= 3) ? std::stoi(argv[2])   : 3;
    const std::string csv_out = (argc >= 4) ? argv[3]              : "pure_cache_results.csv";
    const std::string md_out  = (argc >= 5) ? argv[4]              : "pure_cache_results.md";
    const bool        append  = (argc >= 6 && std::string(argv[5]) == "--append");

#ifdef BENCH_RECORD
    const char* dtype_label = "8B_Record";
#else
    const char* dtype_label = "4B_Int32";
#endif

    // Open Pass A counters: Cycles, Instructions, L1D Read Access, L1D Read Miss
    const int fd_cycles   = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    const int fd_instr    = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    const int fd_l1_load  = open_pmu_counter(
        PERF_TYPE_HW_CACHE,
        make_cache_config(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS)
    );
    const int fd_l1_miss  = open_pmu_counter(
        PERF_TYPE_HW_CACHE,
        make_cache_config(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)
    );

    // Open Pass B counters: LLC Refs, LLC Misses, Branches, Branch Misses
    const int fd_llc_ref  = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    const int fd_llc_miss = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    const int fd_br_tot   = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    const int fd_br_miss  = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

    if (fd_cycles < 0 || fd_instr < 0) {
        std::cerr << "[Error] perf_event_open failed. Run: sudo sysctl -w kernel.perf_event_paranoid=-1\n";
        return 1;
    }

    const std::array<int, 4> pass_a_fds = {fd_cycles, fd_instr, fd_l1_load, fd_l1_miss};
    const std::array<int, 4> pass_b_fds = {fd_llc_ref, fd_llc_miss, fd_br_tot, fd_br_miss};

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
        csv << "Data_Type,Scenario,N,Algorithm,L1_Hit_Rate(%),L1_Miss_Rate(%),LLC_Hit_Rate(%),LLC_Miss_Rate(%),"
               "Branch_Hit_Rate(%),Branch_Miss_Rate(%),IPC,L1_Loads,L1_Misses,LLC_Refs,LLC_Misses,Branches,Branch_Misses,Cycles,Instructions\n";

        md << "# Pure Sorting CPU Cache & Branch Prediction Analysis (N = " << n << ", Repeats = " << repeats << ")\n\n"
           << "> `generate_game_workload()` 및 배열 복사 구간을 100% 제외한 **순수 정렬 함수 내부의 하드웨어 PMU 카운터 측정치**입니다.\n\n"
           << "| Data Type | Scenario | Algorithm | L1 Hit Rate | L1 Miss Rate | LLC Hit Rate | LLC Miss Rate | Branch Hit Rate | Branch Miss Rate | IPC | Cycles | L1 Misses | Branch Misses |\n"
           << "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |\n";
    }

    AlignedBuffer master(n);
    AlignedBuffer work(n);

    for (int s = 0; s < static_cast<int>(Scenario::Count); ++s) {
        const Scenario scenario = static_cast<Scenario>(s);

        // Generate workload ONCE per scenario while all PMU counters are DISABLED
        generate_game_workload(master.data(), n, 1, scenario, 42);

        for (const auto& algo : algos) {
            PurePerfMetrics sum{};

            for (int r = 0; r < repeats; ++r) {
                // --- Pass A: Cycles, Instructions, L1 Loads, L1 Misses ---
                std::memcpy(work.data(), master.data(), n * sizeof(Item));
                for (int fd : pass_a_fds) {
                    if (fd >= 0) {
                        ::ioctl(fd, PERF_EVENT_IOC_RESET, 0);
                        ::ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
                    }
                }

                algo.fn(work.data(), n);

                for (int fd : pass_a_fds) {
                    if (fd >= 0) ::ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
                }

                sum.cycles       += read_pmu_counter(fd_cycles);
                sum.instructions += read_pmu_counter(fd_instr);
                sum.l1_loads     += read_pmu_counter(fd_l1_load);
                sum.l1_misses    += read_pmu_counter(fd_l1_miss);

                // --- Pass B: LLC Refs, LLC Misses, Branches, Branch Misses ---
                std::memcpy(work.data(), master.data(), n * sizeof(Item));
                for (int fd : pass_b_fds) {
                    if (fd >= 0) {
                        ::ioctl(fd, PERF_EVENT_IOC_RESET, 0);
                        ::ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
                    }
                }

                algo.fn(work.data(), n);

                for (int fd : pass_b_fds) {
                    if (fd >= 0) ::ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
                }

                sum.llc_refs      += read_pmu_counter(fd_llc_ref);
                sum.llc_misses    += read_pmu_counter(fd_llc_miss);
                sum.branches      += read_pmu_counter(fd_br_tot);
                sum.branch_misses += read_pmu_counter(fd_br_miss);
            }

            const double inv_r       = 1.0 / static_cast<double>(repeats);
            const double cycles      = sum.cycles        * inv_r;
            const double instr       = sum.instructions  * inv_r;
            const double l1_load     = sum.l1_loads      * inv_r;
            const double l1_miss     = sum.l1_misses     * inv_r;
            const double llc_ref     = sum.llc_refs      * inv_r;
            const double llc_miss    = sum.llc_misses    * inv_r;
            const double br_tot      = sum.branches      * inv_r;
            const double br_miss     = sum.branch_misses * inv_r;

            const double l1_miss_pct  = (l1_load > 0.0) ? (l1_miss / l1_load) * 100.0 : 0.0;
            const double l1_hit_pct   = 100.0 - l1_miss_pct;
            const double llc_miss_pct = (llc_ref > 0.0) ? (llc_miss / llc_ref) * 100.0 : 0.0;
            const double llc_hit_pct  = 100.0 - llc_miss_pct;
            const double br_miss_pct  = (br_tot > 0.0)  ? (br_miss / br_tot) * 100.0 : 0.0;
            const double br_hit_pct   = 100.0 - br_miss_pct;
            const double ipc          = (cycles > 0.0)  ? (instr / cycles) : 0.0;

            char csv_line[512];
            std::snprintf(
                csv_line, sizeof(csv_line),
                "%s,%s,%zu,%s,%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
                dtype_label, scenario_slug(scenario), n, algo.label,
                l1_hit_pct, l1_miss_pct, llc_hit_pct, llc_miss_pct,
                br_hit_pct, br_miss_pct, ipc,
                l1_load, l1_miss, llc_ref, llc_miss, br_tot, br_miss, cycles, instr
            );
            csv << csv_line;

            char md_line[512];
            std::snprintf(
                md_line, sizeof(md_line),
                "| %s | %s | `%s` | **%.2f%%** | %.2f%% | **%.2f%%** | %.2f%% | %.2f%% | **%.2f%%** | %.2f | %.0f | %.0f | %.0f |\n",
                dtype_label, scenario_slug(scenario), algo.label,
                l1_hit_pct, l1_miss_pct, llc_hit_pct, llc_miss_pct,
                br_hit_pct, br_miss_pct, ipc,
                cycles, l1_miss, br_miss
            );
            md << md_line;

            std::printf(
                "%-9s | %-29s | %-22s | %9.2f%% | %9.2f%% | %10.2f%% | %6.2f | %11.0f\n",
                dtype_label, scenario_slug(scenario), algo.label,
                l1_hit_pct, llc_hit_pct, br_miss_pct, ipc, cycles
            );
        }
    }

    for (int fd : pass_a_fds) if (fd >= 0) ::close(fd);
    for (int fd : pass_b_fds) if (fd >= 0) ::close(fd);
    return 0;
#endif
}
