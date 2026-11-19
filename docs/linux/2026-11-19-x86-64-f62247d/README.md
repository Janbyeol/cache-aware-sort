# Linux x86-64 sorting benchmark (2026-11-19)

Source revision `f62247de4469c6d18b4e0d597fa42c612255c18b`.

The repository's elapsed-time suite measured nine sorting algorithms across seven input scenarios and five sizes for both 4-byte integer keys and 8-byte key+ID records. Each CSV contains 315 median results, with per-array median time in microseconds and speedup relative to `std::sort`.

[4-byte integer keys](key_only_4b_summary.csv) · [8-byte key+ID records](key_id_8b_summary.csv)

## Measurement environment

- Intel Core i5-1135G7 with 48 KiB L1D and 1.25 MiB L2 per core, and 8 MiB shared L3.
- Linux x86-64, kernel `7.0.0-43-generic`. Each run was pinned to logical CPU 3 with the `powersave` governor.
- GCC 15.2.0 and Ubuntu libstdc++ package `16-20260322-1ubuntu1`. Makefile flags were `-O3 -std=c++20 -Wall -Wextra -Iinclude`, plus `-DBENCH_RECORD=1` for the 8-byte build.
- Idea 5 used AVX2 for both leaf sorting and merging. Idea 6 selected its AVX2 path at runtime.
- Each result is the median of five repetitions. The runner batches small inputs and times only the sort calls. Input generation, input copying, and correctness checks take place outside the timed interval.

## Idea 6 speedups

Each speedup is the geometric mean across all seven scenarios at that size. Idea 6 was fastest among the nine algorithms in 67 of 70 conditions, and beat both Idea 5 and `std::sort` in all 70.

| Elements | 4B vs. `std::sort` | 4B vs. Idea 5 | 8B vs. `std::sort` | 8B vs. Idea 5 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.47× | 1.79× | 2.34× | 2.06× |
| 2,048 | 7.02× | 1.85× | 3.89× | 2.16× |
| 8,192 | 8.05× | 2.09× | 4.29× | 2.42× |
| 131,072 | 8.09× | 2.11× | 3.78× | 2.26× |
| 1,000,000 | 6.26× | 1.99× | 2.47× | 1.95× |

## One million elements

Median time in milliseconds.

| Scenario | 4B `std::sort` | 4B Idea 5 | 4B Idea 6 | 8B `std::sort` | 8B Idea 5 | 8B Idea 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_3d_distances` | 55.77 | 17.68 | 10.46 | 58.02 | 42.55 | 28.32 |
| `next_frame_motion` | 33.15 | 13.09 | 7.91 | 34.77 | 31.65 | 22.90 |
| `append_5pct_new` | 22.78 | 4.57 | 2.94 | 24.06 | 13.54 | 9.45 |
| `distance_buckets` | 28.78 | 17.66 | 9.04 | 31.05 | 43.22 | 24.09 |
| `mostly_descending` | 5.70 | 3.32 | 0.76 | 6.96 | 10.31 | 1.30 |
| `gaussian_normal` | 49.94 | 17.61 | 10.76 | 52.49 | 43.46 | 29.63 |
| `organ_pipe` | 58.30 | 6.02 | 2.97 | 60.05 | 18.20 | 10.50 |

At this size, Idea 6 was fastest in 13 of 14 conditions. Idea 3 led the 8-byte `append_5pct_new` scenario at 6.80 ms.

## Correctness

On x86-64, both unit-test binaries passed 4,294 checks each. Each format also passed the same suite with Idea 5 using the scalar path and the SSE2, AVX2, and AVX-512F backends in leaf-only, merge-only, and combined modes. Both benchmark runs verified every timed result against the reference sort, including key+ID integrity in the 8-byte run.

On Linux AArch64 under QEMU user-mode 10.2.1 (`-cpu max`), binaries cross-compiled with GCC 15.2.0 passed 4,294 unit checks per format and completed both full benchmark suites with NEON. The published elapsed times come from native x86-64 runs.

## Reproduction

Build from the source revision above and run the elapsed-time suite in the directory where you want the CSV files.

```bash
make -f makefile.txt all
export HSORT_SIMD_ISA=avx2 HSORT_SIMD_STAGE=both
taskset -c 3 ./test_4b
taskset -c 3 ./test_8b
taskset -c 3 ./bench_4b
taskset -c 3 ./bench_8b
```
