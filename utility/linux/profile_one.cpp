#include "hybrid_sorts.hpp"
#include <cstring>
#include <iostream>
#include <string>

#ifdef __linux__
#include <sys/prctl.h>
#define PERF_DISABLE() prctl(PR_TASK_PERF_EVENTS_DISABLE, 0, 0, 0, 0)
#define PERF_ENABLE()  prctl(PR_TASK_PERF_EVENTS_ENABLE, 0, 0, 0, 0)
#else
#define PERF_DISABLE() ((void)0)
#define PERF_ENABLE()  ((void)0)
#endif

using namespace hsort;

int main(int argc, char** argv) {
    // 1. Pause hardware performance counters during memory allocation & data generation
    PERF_DISABLE();

    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " <algo: std|stable|idea1|idea2|idea3|idea4> <scenario: 0..6> [N=1000000]\n";
        return 1;
    }

    const std::string algo = argv[1];
    const int scen_id      = std::stoi(argv[2]);
    const std::size_t n    = (argc >= 4) ? std::stoull(argv[3]) : 1000000;

    AlignedBuffer data(n);
    generate_game_workload(data.data(), n, 1, static_cast<Scenario>(scen_id), 42);

    // 2. Enable hardware performance counters strictly for the sorting algorithm
    PERF_ENABLE();

    if      (algo == "std"    || algo == "std::sort")              std_sort_wrapper(data.data(), n);
    else if (algo == "stable" || algo == "std::stable_sort")       std_stable_sort_wrapper(data.data(), n);
    else if (algo == "idea1"  || algo == "cache_leaf_quick")       cache_leaf_quick_sort(data.data(), n);
    else if (algo == "idea2"  || algo == "mean_pivot_quick")       mean_pivot_quick_sort(data.data(), n);
    else if (algo == "idea3"  || algo == "boundary_trimmed_merge") boundary_trimmed_merge_sort(data.data(), n);
    else if (algo == "idea4"  || algo == "tiled_pingpong_merge")   tiled_pingpong_merge_sort(data.data(), n);
    else {
        PERF_DISABLE();
        std::cerr << "Unknown algo: " << algo << "\n";
        return 1;
    }

    // 3. Immediately disable hardware performance counters after sorting completes
    PERF_DISABLE();
    return 0;
}
