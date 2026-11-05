# Cache-Aware Hybrid Sort Study (64B Cache-Line Optimization)

**English** | [한국어](README.ko.md)

A C++20 experiment in small-block sorting, boundary trimming, and tiled merging. The project compares four custom sorting variants with `std::sort`, `std::stable_sort`, and a basic Quicksort on synthetic, game-inspired workloads.

The focus is on **4-byte integer keys** and **8-byte key + entity ID records**. This is a learning project with workload-dependent results, not a general-purpose replacement for the standard library.

> **AI collaboration**
>
> I am an undergraduate studying algorithms and computer architecture. I heavily relied on ChatGPT and Gemini to implement the C++20 code, explore low-level optimizations, and build the benchmark/profiling tools. The initial hypotheses came from my own questions while studying sorting. That describes how the project started, not a claim that these techniques are new.

## Contents

- [Motivation](#motivation)
- [Build and run](#build-and-run)
- [How the four ideas evolved](#how-the-four-ideas-evolved)
- [Workloads](#workloads)
- [Recorded results](#recorded-results)
- [Benchmark methodology and reproduction](#benchmark-methodology-and-reproduction)
- [Scope and limitations](#scope-and-limitations)
- [Repository guide](#repository-guide)
- [Feedback and contributions](#feedback-and-contributions)

## Motivation

While studying sorting algorithms, I wondered: *Quicksort has an O(N²) worst case and the same O(N log N) average complexity as Heapsort or Mergesort. Why can it be faster in practice?*

Memory access patterns, comparison and movement costs, branch behavior, and implementation details all matter. That led to two questions:

1. Would switching to insertion sort at a **64-byte leaf size** reduce the overhead of sorting tiny partitions?
2. Could partition statistics or searches outward from a merge boundary reduce unnecessary work?

The implementation uses a fixed 64-byte leaf budget and 8 KiB tiles. A leaf therefore contains 16 `int32_t` keys or 8 records. These sizes are tuning choices: a 64-byte range is not necessarily cache-line aligned, and its size does not guarantee an L1 hit or cache residency.

## Build and run

### Linux / GCC

Requirements: a C++20-capable `g++`, GNU Make, and a shell. Run from the repository root:

```bash
git clone https://github.com/Janbyeol/cache-aware-sort.git
cd cache-aware-sort

# Build and run correctness tests for both item formats.
make -f makefile.txt test

# Build all four test/benchmark executables.
make -f makefile.txt all

# Run the wall-clock benchmark for both formats.
make -f makefile.txt bench
```

The filename is **`makefile.txt`**, so the `-f` option is required. The default flags are `-O3 -march=native -std=c++20 -Wall -Wextra -Iinclude`; binaries built with `-march=native` may not run on a different CPU.

- `test_4b` / `test_8b`: correctness checks. The current suite performs **3,372 checks per format**, covering boundary sizes, all seven workloads, randomized cases, record integrity, allocation overflow, and additional boundary-trimmed merge regressions.
- `bench_4b` / `bench_8b`: fixed-size wall-clock benchmark runners. They write `key_only_4b_summary.csv` and `key_id_8b_summary.csv` in the working directory.
- `make -f makefile.txt bench` runs the complete suite, not a single-size benchmark. Use the platform profiling scripts below to choose `N`.

### Windows / Visual Studio

With Visual Studio C++ build tools and a Windows SDK installed:

```powershell
.\generate_solution.bat
```

Open the generated `cache-aware-sort.sln`, select **Release | x64**, and build. The solution contains `test_4b`, `test_8b`, `bench_4b`, and `bench_8b`; executables are written to `bin\Release\`.

```powershell
.\bin\Release\test_4b.exe
.\bin\Release\test_8b.exe
```

The generated Release projects enable AVX2, so they require a compatible x64 target. The solution generator is **not** the native ARM64 build path. For Windows ARM64 profiling, use the separate scripts described below.

## How the four ideas evolved

The implementations share the item types and low-level helpers in [`include/hybrid_sorts.hpp`](include/hybrid_sorts.hpp).

### Idea 1: Cache-line-sized Quicksort leaves

[`cache_leaf_quick`](src/idea1_cache_leaf_quick.cpp) uses median-of-three Hoare partitioning, then switches to insertion sort when a partition fits within the 64-byte leaf budget. The experiment isolates the cost of continuing Quicksort down to one element: the `pure_quick` baseline uses the same partitioning helper without the larger cutoff.

The cutoff reduces partitioning work on small ranges, but it does not remove Quicksort's data-dependent branches or its worst-case behavior.

### Idea 2: Mean-pivot prediction

[`mean_pivot_quick`](src/idea2_mean_pivot_quick.cpp) performs Hoare partitioning and then **scans both partitions again** to collect sums and minimum/maximum keys. Those statistics choose subsequent mean pivots and let the algorithm skip all-equal ranges.

The current implementation does not collect those statistics during the partition pass. The extra scans are part of the trade-off. Mean pivots are clamped to ensure progress when integer division, especially with negative keys, would otherwise choose the maximum key.

### Idea 3: Boundary-trimmed bottom-up merge

[`boundary_trimmed_merge`](src/idea3_boundary_trimmed_merge.cpp) changes the starting point from top-down partitioning to bottom-up merging:

1. Sort 64-byte leaves with insertion sort.
2. Complete the early merge passes within each 8 KiB tile.
3. Skip a merge if the two runs are already ordered at their boundary.
4. Otherwise, search outward from the boundary to find the overlapping middle range. Short linear scans are followed by boundary checks, exponential search, and binary search.
5. For a small or strongly asymmetric overlap, copy only the smaller side to scratch space and merge back directly. For a larger balanced overlap, use a bidirectional merge into scratch space, then copy the result back.

This is the most useful direction in the recorded append-tail and jittered workloads. It also has substantial regressions on some other inputs, shown below.

### Idea 4: Tiled ping-pong merge

[`tiled_pingpong_merge`](src/idea4_tiled_pingpong_merge.cpp) also uses 8 KiB tiles, but alternates source and destination buffers between merge passes. It uses binary searches to trim the overlap and copies the already ordered outer ranges to the destination when needed.

It also probes global ordering trends, adjusts mostly descending input, and confirms sortedness before taking an early return. The probe alone is not treated as proof that the array is sorted.

Both merge variants use conditional selection in their merge kernels. AI helped implement these kernels, boundary searches, and tiling. Whether the compiler emits `cmov`, conditional-select instructions, or branches depends on the target and compiler; the algorithms are not entirely branch-free.

## Workloads

All inputs are generated by [`generate_game_workload`](src/benchmark_api.cpp). They are synthetic workloads, not measurements from a running game engine.

| ID | Workload | What is generated |
| --- | --- | --- |
| 0 | Random 3D distances | Squared-distance keys from pseudo-random 3D coordinates. The resulting keys are **not uniformly distributed**, despite the profiler's legacy `0_Uniform_Random` label. |
| 1 | Frame jitter | Sort points by distance, perturb their coordinates, then recompute distance keys. |
| 2 | Append 5% tail | A sorted prefix followed by `floor(N / 20)` unsorted new keys. |
| 3 | Distance buckets | Divide distance keys by 1,000,000 to create roughly 300 buckets and many duplicates. |
| 4 | Mostly descending | Descending keys perturbed by approximately 1% random adjacent swaps. |
| 5 | Gaussian-like | Sum four pseudo-random integer samples to obtain a bell-shaped distribution, not an exact normal distribution. |
| 6 | Organ pipe | Arrange sorted keys into an ascending then descending mountain shape. |

## Recorded results

The committed [Linux CSVs](docs/linux/) contain **hardware counters**, not wall-clock runtimes. The tables below use the [`N = 1,000,000` CSV](docs/linux/pure_cache_results_N=1000000.csv), with **4-byte integer keys**. They report the stored measurements rather than a fresh run of the current checkout.

The CSVs do not include the original CPU model, compiler/version, operating-system details, exact build revision, or per-trial samples. Treat them as an exploratory snapshot, not a portable performance guarantee or a fully reproducible benchmark record. Millisecond values from earlier README drafts are omitted because the committed CSVs do not substantiate them.

### Cycle counts across all seven scenarios

`M` means one million cycles. Parentheses show `std::sort cycles / algorithm cycles`: a ratio above 1 means fewer recorded cycles than `std::sort`, **not** an equivalent wall-clock speedup.

| Workload | `std::sort` | `std::stable_sort` | Boundary-trimmed merge | Tiled ping-pong merge |
| --- | ---: | ---: | ---: | ---: |
| Random 3D distances | 230.79M | 278.63M | 142.74M (1.62×) | 139.40M (1.66×) |
| Frame jitter | 137.26M | 150.98M | 90.97M (1.51×) | 97.25M (1.41×) |
| Append 5% tail | 85.48M | 38.75M | 24.66M (3.47×) | 28.76M (2.97×) |
| Distance buckets | 114.67M | 164.06M | 142.81M (0.80×) | 139.54M (0.82×) |
| Mostly descending | 23.68M | 40.15M | 110.38M (0.21×) | 17.14M (1.38×) |
| Gaussian-like | 207.85M | 259.24M | 142.79M (1.46×) | 139.30M (1.49×) |
| Organ pipe | 246.65M | 34.95M | 61.74M (3.99×) | 61.05M (4.04×) |

The counterexamples matter: both custom merge variants use more cycles than `std::sort` on duplicate-heavy distance buckets, and boundary-trimmed merge uses about **4.66× as many cycles** on mostly descending input. `std::stable_sort` beats both on organ-pipe input. There is no single winner across these workloads.

### Cache and branch counters do not tell the same story

For the random 3D-distance workload above:

| Algorithm | IPC | L1D read hit rate | L1D read misses | Branch prediction hit rate |
| --- | ---: | ---: | ---: | ---: |
| `std::sort` | 0.63 | 97.83% | 662,145 | 75.54% |
| `boundary_trimmed_merge` | 2.19 | 95.11% | 2,020,305 | 95.02% |
| `tiled_pingpong_merge` | 2.21 | 96.93% | 1,256,029 | 94.89% |

The merge variants have fewer recorded cycles, higher IPC, and better branch prediction hit rates here, but **more L1D read misses and lower L1D read hit rates**. These observations do not establish improved cache behavior as the cause of the cycle reduction. Hit rates also have different access-count denominators, so compare absolute counts as well as percentages.

The ten committed CSVs cover both formats at `N = 16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000`. Evaluate 8-byte records and other sizes separately: for example, the 8-byte ping-pong variant uses about 16.3% more cycles than `std::sort` on the one-million-item mostly descending workload, unlike the 4-byte result above. [`docs/windowsARM/`](docs/windowsARM/) currently contains a placeholder, not published measurements.

## Benchmark methodology and reproduction

### Wall-clock suite

`make -f makefile.txt bench` runs seven algorithms, seven scenarios, and five sizes (`128, 2048, 8192, 131072, 1000000`). It reports the median of five per-array timing samples, batching smaller arrays to reduce timer overhead. Input generation, the input copy, and result verification are outside the timed interval; work performed inside the sort, including scratch allocation and internal copies, remains included.

This runner verifies sorted keys after each timed sample and checks record IDs/payloads in 8-byte mode. Its `trimmed_merge` output label refers to `boundary_trimmed_merge_sort`.

### Linux hardware counters

Use Linux with GCC, Bash, and hardware performance-counter access already permitted by the host. The pure profiler calls `perf_event_open` directly. The separate `run_perf_csv.sh` wrapper uses `perf stat` around the whole process, so its measurement scope differs.

```bash
# One size, three repetitions, both 4B and 8B formats.
bash utility/linux/run_pure_perf.sh 1000000 3 results.csv

# All ten sizes, three repetitions, five seconds between sizes.
# Use a separate output directory to preserve the committed measurements.
bash utility/linux/run_pure_perf.sh --all 3 5 results/linux
```

The scripts compile the profiling binaries and write CSV/Markdown summaries; the full sweep also writes `pure_cache_results_all.csv`.

The pure profiler reuses seed `42`, copies the source before each measured sort, collects counters in two separate passes, and averages counter values across repetitions. It excludes input generation and the pre-sort copy, but includes sort-internal allocation/copies and some counter-control overhead. The copy affects cache state; this is not a controlled cold-cache experiment. It does **not** perform the wall-clock runner's result verification, so run the correctness suite first.

Counter support and permissions vary by CPU, kernel, VM, and container. A failed cycle/instruction counter open stops the profiler; unavailable optional counters can appear as zero counts and misleading 100% hit rates. Check counter availability before interpreting results. Events are enabled individually and may be multiplexed/scaled, so short runs and cross-event ratios need particular care. The generic cache events labelled `LLC` have CPU-dependent meanings. CPU affinity is not pinned by the Linux pure profiler.

If access is denied, use an environment with appropriate PMU access or consult its administrator. Do not assume that changing system-wide security settings is required or appropriate. See the [Linux profiling guide](utility/linux/LINUX_PERF_GUIDE.md) for script details.

### Windows ARM64 counters and timing

The Windows profiler uses `QueryPerformanceCounter` for elapsed time and `QueryThreadCycleTime` for thread cycles. These are **not** Linux L1D/LLC/branch counters and should not be compared as if the metric sets were identical. Its warmup, small-input batching, and median aggregation also differ from the Linux pure profiler.

```powershell
# One size, three repetitions, both formats.
.\utility\windowsARM\run_pure_perf.bat 1000000 3 results.csv

# Full sweep into a separate output directory.
.\utility\windowsARM\run_pure_perf.bat --all 3 5 results\windowsARM
```

The script prefers Visual Studio's compiler and selects ARM64 on an ARM64 host, with Clang/GCC fallbacks. Install the relevant C++ toolchain and SDK, and verify the reported architecture rather than assuming every fallback produces a native ARM64 binary. See the [Windows ARM64 guide](utility/windowsARM/WINDOWS_ARM_GUIDE.md).

For new benchmark reports, include the commit SHA, CPU/cache configuration, OS, compiler and standard-library versions, flags, affinity/power settings, counter availability, item format, `N`, workload, seed, and repeated measurements. Keep wall-clock timings and hardware-counter runs clearly separated.

## Scope and limitations

- **API:** The functions accept `hsort::Item*` and an element count. They sort ascending by the signed 32-bit key. The build chooses either `int32_t` or `DistanceRecord { key, entity_id }` with `-DBENCH_RECORD=1`; there is no generic iterator/comparator interface.
- **Memory:** For non-leaf inputs, boundary-trimmed merge allocates scratch space for `N` items. Tiled ping-pong merge does the same when its global-trend fast path does not finish the sort. Half-buffer merging reduces copying for selected overlaps, not the allocated capacity. Allocation can fail.
- **Complexity:** The merge variants have O(N log N) worst-case work and O(N) auxiliary storage. The custom Quicksort variants have no introspective fallback and can take O(N²) work on unfavorable inputs.
- **Stability:** Boundary-trimmed merge preserves equal-key order, with dedicated 8-byte record regressions against `std::stable_sort`. The custom Quicksort variants are unstable. Tiled ping-pong merge is also **not guaranteed stable** because its global descending-input adjustment can reorder equal-key records.
- **Portability:** The 64-byte and 8 KiB constants are fixed, not detected from the machine. Generated instructions and performance depend on the CPU, compiler, flags, standard-library implementation, and workload.
- **Evidence:** Synthetic workloads and a passing test suite do not establish production readiness. The benchmarks do not cover arbitrary user-defined comparators, larger records, parallel sorting, or every adversarial input.

My takeaway is that boundary trimming is worth exploring when much of the input is already ordered, while ping-pong merging offers a different copying trade-off. The useful next step is to measure the actual workload and explain both the wins and the losses, rather than assuming that a cache-sized block makes a sort universally faster.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`include/hybrid_sorts.hpp`](include/hybrid_sorts.hpp) | Item types, constants, aligned buffer, shared kernels, and public declarations |
| [`src/baseline_sorts.cpp`](src/baseline_sorts.cpp) | Basic Quicksort and standard-library baselines |
| [`src/`](src/) | Four custom implementations, workload generator, correctness tests, and timing runner |
| [`makefile.txt`](makefile.txt) | GCC test and benchmark targets |
| [`generate_solution.bat`](generate_solution.bat) | Visual Studio x64 solution generator |
| [`utility/linux/`](utility/linux/) | Linux PMU profiler and external `perf stat` wrapper |
| [`utility/windowsARM/`](utility/windowsARM/) | Windows timing/thread-cycle profiler and build scripts |
| [`docs/linux/`](docs/linux/) | Committed Linux counter CSVs and plots |
| [`docs/windowsARM/`](docs/windowsARM/) | Placeholder for Windows ARM64 results |

## Feedback and contributions

Please roast my code, but tell me why! I am still learning, and feedback on correctness, algorithm design, benchmarking, and low-level implementation is welcome.

Open an [issue](https://github.com/Janbyeol/cache-aware-sort/issues) or submit a pull request. For a bug, include a small failing input and the item format. For a performance claim, include the environment, exact command, and raw results so the comparison can be checked. References to related sorting techniques are welcome too.

## License

This project is licensed under the [MIT License](LICENSE).
