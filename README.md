# Cache-Aware Hybrid Sort Study (64B Cache-Line Optimization)

**English** | [한국어](README.ko.md)

A C++20 experiment in small-block sorting, boundary trimming, and tiled merging. The initial study compares four custom sorting variants (Ideas 1–4) with `std::sort`, `std::stable_sort`, and a basic Quicksort on synthetic, game-inspired workloads, with subsequent follow-up experiments (Idea 5+) explored separately.

The focus is on **4-byte integer keys** and **8-byte key + entity ID records**. This is a learning project with workload-dependent results, not a general-purpose replacement for the standard library.

> **AI collaboration**
>
> I am an undergraduate studying algorithms and computer architecture. I heavily relied on ChatGPT and Gemini to implement the C++20 code, explore low-level optimizations, and build the benchmark/profiling tools. The initial hypotheses came from my own questions while studying sorting. That describes how the project started, not a claim that these techniques are new.

## Contents

- [Motivation](#motivation)
- [Build and run](#build-and-run)
- [How the four ideas evolved](#how-the-four-ideas-evolved)
- [Follow-up ideas (Idea 5+)](#follow-up-ideas-idea-5)
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

Use `-f makefile.txt`. The default build flags are `-O3 -std=c++20 -Wall -Wextra -Iinclude`.

- `test_4b` / `test_8b`: correctness checks. The suite covers boundary sizes, all seven workloads, randomized cases, record integrity, allocation overflow, and additional boundary-trimmed merge regressions.
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

The solution builds x64 executables with an SSE2 baseline. Use the separate Windows ARM64 profiling scripts below for native ARM64 builds.

## How the four ideas evolved

The initial implementation explored four ideas sharing the item types and low-level helpers in [`include/hybrid_sorts.hpp`](include/hybrid_sorts.hpp).

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

## Follow-up ideas (Idea 5+)

The initial implementation and the archived Linux hardware-counter study below concluded with Ideas 1–4. The ideas in this section were added later as follow-up experiments sparked by questions that came up after finishing Idea 4.

### Idea 5: SIMD tiled ping-pong merge

After finishing Idea 4, a natural follow-up question came up: since each 64-byte leaf holds a fixed number of elements (16× 4B keys or 8× 8B records), what happens if we vectorize the 64-byte leaf sort and the overlapping merge step using SIMD bitonic sorting networks?

[`simd_tiled_pingpong_merge_sort`](src/idea5_simd_tiled_merge.cpp) reuses Idea 4's 8 KiB tiles, binary boundary trimming, and ping-pong buffer swaps, while replacing full 64-byte leaves and complete vector overlaps with a bitonic network ([`src/simd/network.hpp`](src/simd/network.hpp)). For 8-byte records, comparisons use the signed 32-bit key and move the 32-bit entity ID alongside it. Equal keys can change order.

Set `HSORT_SIMD_ISA` before launch to `auto` (default), `scalar`, `sse2`, `avx2`, `avx512f`, or `neon`. Auto currently chooses AVX2 or SSE2 on x64, NEON on supported ARM64 builds, and scalar elsewhere. Select `avx512f` to use the shared 512-bit kernel on AVX-512F and compatible AVX10.2 CPUs. `HSORT_SIMD_STAGE` accepts `leaf`, `merge`, and `both` (default) so the leaf and merge vectorization stages can be isolated independently. The profiler label records the selected kernel, active stages, and detected CPU capability. See the [Windows x64 AVX2 follow-up benchmark](docs/windows-x64/2026-11-19-avx2/README.md) for full comparisons across all eight sorts and stage-by-stage isolation of SIMD leaf vs. SIMD merge.

### Idea 6: Transpose-vectorized quad-leaf & dual-chain branchless SIMD tiled merge

[`simd_transpose_tiled_merge_sort`](src/idea6_simd_transpose_merge.cpp) (`simd_transpose_merge`) addresses the bottlenecks observed in Idea 5 (cross-lane permute latency, serial loop-carried `carry` dependency, 8B record comparison overhead, and dirtying clean cache lines on pre-sorted blocks) by lifting the core principles of Ideas 1, 3, and 4 directly into SIMD register architecture:

1. **Vertical Sorting Network + In-Register Matrix Transpose for 256B Quad-Leaves (`4 * LEAF_SIZE`)**: Loads 4 cache lines (8 YMM registers: 64× 4B keys or 32× 8B records), sorts columns vertically with zero shuffles, transposes the register matrix ($8 \times 8$ or $4 \times 4 \times 2$), and finishes merging the 256B quad-leaf in registers—eliminating the first two intra-tile merge passes (`width = LEAF_SIZE, 2 * LEAF_SIZE`) while keeping the remaining 5 intra-tile passes odd.
2. **1-Cycle In-Lane Immediate Shuffles & Single-Comparison Tie-Safe 8B Kernel**: Replaces 3-cycle cross-lane variable permutes (`vpermd`) with 1-cycle in-lane `_mm256_shuffle_epi32` and immediate blends, and compares 8B records (`{ key, entity_id }`) with a single `_mm256_cmpgt_epi32` per stage while preserving equal-key records. (On ARM64 NEON, it uses hardware `vld2q_s32`/`vst2q_s32` key/ID de-interleaving for 4-lane leaf sorting paired with 64-bit `csel` bidirectional merge.)
3. **Dual-Chain (`carry_fwd` + `carry_bwd`) Branchless SIMD Bitonic Merge**: Extends Idea 3's bidirectional merge to SIMD bitonic merging by advancing two independent SIMD merge chains from the front and back simultaneously with branchless `cmov` pointer updates.
4. **Clean Cache-Line Preservation, `O(1)` Outer-Boundary Check & Run-Boundary Short-Circuiting**: Skips memory stores on already-sorted 256B quad-leaves to avoid dirtying cache lines on nearly-sorted inputs, combines Idea 3's `O(1)` outer-boundary check with binary boundary trimming, and short-circuits merge passes when all run boundaries are ordered.
5. **Stack-Resident L1 Scratch for `N <= TILE_SIZE` (8 KiB) & Single-TU Inlining**: Avoids heap allocation on small/medium arrays and eliminates Windows x64 non-volatile `xmm6`–`xmm15` spill overhead across leaf and merge calls.

#### Idea 6 Head-to-Head Results (`N = 1,000,000`, Windows x64 · Intel Core Ultra 7 270K Plus · `g++ -O3`, median ms)

**4-Byte Integer Keys (`key_only_4b`)**

| Workload | `std::sort` | `Idea 3 (Trim)` | `Idea 4 (Tiled)` | `Idea 5 (SIMD)` | **Idea 6** | vs. `std::sort` | vs. `Idea 4` | vs. `Idea 5` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random 3D distances | 45.54 ms | 24.51 ms | 22.56 ms | 13.14 ms | **6.76 ms** | **6.73×** | **3.33×** | **1.94×** |
| Frame jitter | 26.24 ms | 15.62 ms | 16.64 ms | 10.32 ms | **5.90 ms** | **4.45×** | **2.82×** | **1.75×** |
| Append 5% tail | 17.50 ms | 3.76 ms | 4.57 ms | 3.63 ms | **3.06 ms** | **5.71×** | **1.49×** | **1.19×** |
| Distance buckets | 22.88 ms | 24.42 ms | 22.27 ms | 13.10 ms | **6.55 ms** | **3.49×** | **3.40×** | **2.00×** |
| Mostly descending | 3.30 ms | 19.30 ms | 2.98 ms | 2.82 ms | **2.23 ms** | **1.48×** | **1.34×** | **1.27×** |
| Gaussian-like | 40.68 ms | 24.29 ms | 22.79 ms | 13.33 ms | **6.98 ms** | **5.83×** | **3.27×** | **1.91×** |
| Organ pipe | 39.97 ms | 11.37 ms | 9.24 ms | 4.48 ms | **3.86 ms** | **10.34×** | **2.39×** | **1.16×** |

**8-Byte Key + Entity ID Records (`key_id_8b`)**

| Workload | `std::sort` | `Idea 3 (Trim)` | `Idea 4 (Tiled)` | `Idea 5 (SIMD)` | **Idea 6** | vs. `std::sort` | vs. `Idea 4` | vs. `Idea 5` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random 3D distances | 43.85 ms | 32.25 ms | 29.32 ms | 32.54 ms | **18.19 ms** | **2.41×** | **1.61×** | **1.79×** |
| Frame jitter | 25.90 ms | 19.05 ms | 21.99 ms | 21.62 ms | **14.00 ms** | **1.85×** | **1.57×** | **1.54×** |
| Append 5% tail | 17.46 ms | 4.74 ms | 8.33 ms | 9.18 ms | **6.56 ms** | **2.66×** | **1.27×** | **1.40×** |
| Distance buckets | 22.70 ms | 32.63 ms | 29.56 ms | 33.21 ms | **18.30 ms** | **1.24×** | **1.62×** | **1.81×** |
| Mostly descending | 4.33 ms | 22.67 ms | 6.01 ms | 6.82 ms | **4.47 ms** | **0.97×** | **1.34×** | **1.53×** |
| Gaussian-like | 39.14 ms | 32.21 ms | 29.85 ms | 33.04 ms | **18.75 ms** | **2.09×** | **1.59×** | **1.76×** |
| Organ pipe | 40.33 ms | 12.45 ms | 13.07 ms | 13.67 ms | **10.75 ms** | **3.75×** | **1.22×** | **1.27×** |

See the **[Idea 6 Design and Benchmark Guide](docs/idea6-simd-transpose/README.md)** for full architectural diagrams, microarchitectural analysis, and multi-scale (`N = 128 .. 1,000,000`) benchmark tables.

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

`make -f makefile.txt bench` runs nine algorithms, seven scenarios, and five sizes (`128, 2048, 8192, 131072, 1000000`). It reports the median of five per-array timing samples, batching smaller arrays to reduce timer overhead. Input generation, the input copy, and result verification are outside the timed interval; work performed inside the sort, including scratch allocation and internal copies, remains included.

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
| [`src/`](src/) | Initial four custom implementations (Ideas 1–4), follow-up SIMD implementations (Idea 5+), workload generator, correctness tests, and timing runner |
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
