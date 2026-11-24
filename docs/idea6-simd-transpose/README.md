# Idea 6: Transpose-Vectorized Quad-Leaf & Dual-Chain Branchless SIMD Tiled Merge Sort (AVX2 & ARM64 NEON)

[한국어](README.ko.md) | **English**

This document details the hypothesis, assembly-level microarchitectural and compiler bottleneck analysis, architectural mechanisms, four-iteration ARM64 NEON optimization progression (`Baseline → v1 → v2 → v3 → v4`), and head-to-head benchmark results of **Idea 6 (`simd_transpose_merge` / `simd_transpose_tiled_merge_sort`)** across both **x86-64 AVX2 (256-bit)** and **Windows ARM64 NEON (128-bit, Snapdragon X Elite / Oryon)**.

---

## Contents
1. [Motivation and Evolution from Ideas 1–5](#1-motivation-and-evolution-from-ideas-15)
2. [Microarchitectural & Compiler Bottleneck Analysis (AVX2 & ARM64 NEON)](#2-microarchitectural--compiler-bottleneck-analysis-avx2--arm64-neon)
3. [Core Architectural Pillars of Idea 6](#3-core-architectural-pillars-of-idea-6)
4. [ARM64 NEON Iterative Engineering Log (Baseline → v1 → v2 → v3 → v4)](#4-arm64-neon-iterative-engineering-log-baseline--v1--v2--v3--v4)
5. [Head-to-Head Benchmark Results (Windows ARM64 NEON & Windows x64 AVX2)](#5-head-to-head-benchmark-results-windows-arm64-neon--windows-x64-avx2)

---

## 1. Motivation and Evolution from Ideas 1–5

1. **Idea 1 (64B Cache-Line Leaf Cutoff)**: Stopping partitioning at 64 bytes (`LEAF_SIZE`: 16× 4B keys or 8× 8B records) removes recursion and branch overhead on tiny subarrays.
2. **Idea 3 (Bidirectional Branchless Merge + `O(1)` Outer-Boundary Check)**:
   - Merging from the front (`->`, minimum selection) and back (`<-`, maximum selection) simultaneously splits the loop-carried dependency chain into two independent chains, doubling instruction-level parallelism (ILP).
   - Checking `data[begin] > right_min` and `data[end - 1] < left_max` in `O(1)` before binary searches avoids redundant boundary probing on random data.
3. **Idea 4 (Global Trend Probe + 8 KiB L1 Tiling + Ping-Pong Merge)**:
   - Completing leaf sorts and early merge passes inside 8 KiB L1 tiles (`TILE_SIZE`) while alternating source and destination buffers eliminates per-pass copy-back overhead.
4. **Idea 5 (Per-Leaf Horizontal Bitonic Network + Single-Chain Vector Merge)**:
   - Vectorized individual 64B leaves and overlap merges using horizontal variable permutes (`vpermd` / NEON shuffles) and a single-direction `carry` loop.
   - However, **8B records (`{ key, entity_id }`) on AVX2 regressed behind scalar `Idea 4`**, and **on Windows ARM64 NEON (Snapdragon X Elite), both 4B and 8B were slower than `std::sort`**.

---

## 2. Microarchitectural & Compiler Bottleneck Analysis (AVX2 & ARM64 NEON)

### Bottleneck ①: Cross-Lane Permute Overhead in Horizontal Per-Leaf Bitonic Sorting
- **AVX2**: Calling `_mm256_permutevar8x32_epi32` (`vpermd`) at every stage of a horizontal bitonic network incurs a **3-cycle latency on a single execution port (Port 5)** plus constant-mask register pressure.
- **ARM64 NEON**: Disassembling `neon.s` showed `Network<Neon>::full_leaf` took **166 NEON instructions per 16-element (64B) leaf** due to repeated horizontal lane shuffles across four 128-bit registers.

### Bottleneck ②: Single-Chain Loop-Carried `carry` Dependency (21-Cycle Latency Floor on NEON)
In a single-direction SIMD bitonic merge loop:

```math
\text{merge\_vec}(\text{next\_vec}, \text{carry}) \rightarrow (\text{out\_low}, \text{new\_carry})
```

each iteration cannot begin until the previous iteration's `new_carry` exits the full bitonic network.
- **ARM64 NEON Assembly Verification**: In `neon.s`, `Network<Neon>::merge` had a serial `carry` dependency chain of **11 NEON instructions (\~21 cycles latency)** per 4 elements (16B): 2 instructions for vector reversal (`rev64` + `ext`) and 9 instructions for two bitonic butterfly stages.
- Across 16 merge passes at `N = 1,000,000` ($4,000,000$ vector merges), this imposed a hard theoretical latency floor of $4\text{M} \times 21 = 84\text{M}$ cycles, matching measured hardware cycles (**85.15M cycles / 28.82 ms**).

### Bottleneck ③: LLVM/Clang AArch64 De-If-Conversion (`CodeGenPrepare` `select-opt`) Trap
- In `include/hybrid_sorts.hpp`, `bidirectional_branchless_merge` was written in branchless C++ (`l_fwd += !take_right; r_fwd += take_right;`).
- Inspecting the generated ARM64 assembly (`idea4.s` lines 659–685, `neon.s` lines 230–258) revealed that **LLVM's AArch64 backend (`aarch64-enable-select-opt`) secretly converted those branchless pointer increments back into two data-dependent conditional branches (`b.ge` and `b.le`) per loop iteration**:
  ```armasm
  // idea4.s (lines 659-685) - Clang de-if-converting branchless C++ back into conditional branches
  .LBB4_22:
      ldr   w16, [x14]          // *r_fwd
      ldr   w17, [x13]          // *l_fwd
      cmp   w16, w17
      csel  w16, w16, w17, lt   // Uses csel for value selection...
      b.ge  .LBB4_24            // [CRITICAL] Emits 50%-mispredicted branch for forward pointer!
      mov   x16, x13
      b     .LBB4_25
  .LBB4_24:
      add   x16, x13, #4
  .LBB4_25:
      ...
      cmp   w18, w17
      b.le  .LBB4_27            // [CRITICAL] Emits second 50%-mispredicted branch for backward pointer!
  ```
- On random data, two 50%-mispredicted branches per element pair inflated scalar merge runtime on Snapdragon X Elite to **66.54 ms (185.4M cycles)**.

### Bottleneck ④: 128-bit NEON 2-Lane AoS Penalty & AVX2 Dual-Comparison Cost on 8B Records
- **AVX2**: Without a hardware 64-bit key-only `min/max`, comparing `{ key, id }` records separately for `min` and `max` doubles `vpcmpgtd`, `vpshufd`, and `vpblendvb` instructions.
- **ARM64 NEON**: A 128-bit NEON register holds only **2 records (16B)** in AoS layout. In Idea 5 (`src/simd/neon.cpp`), emulating immediate blends required loading 16B masks from memory (`vld1q_u32`) + `vbslq_s32` at every step, taking **51.21 ms (150.2M cycles)**.

### Bottleneck ⑤: Dirtying Clean Cache Lines & 8 MB Scratch Page-Fault Overhead on `4_Mostly_Descending`
- Unconditionally storing SIMD vectors on pre-sorted blocks transitions Clean cache lines to Modified (Dirty), amplifying writeback traffic.
- On `4_Mostly_Descending` (which has \~1% local noise after reversing the array), falling through to full ping-pong merge allocates an 8 MB `scratch` buffer, incurring **2,048 OS zero-page soft page faults (\~3.5 ms)** and 16 MB of buffer copying.

---

## 3. Core Architectural Pillars of Idea 6

Implemented in [`src/idea6_simd_transpose_merge.cpp`](../../src/idea6_simd_transpose_merge.cpp):

1. **Vertical Sorting Network + In-Register Matrix Transpose for 256B Quad-Leaves (`sort_quad_leaf_avx2` / `sort_quad_leaf_neon`)**:
   - Loads four cache lines (`4 * LEAF_SIZE = 256B`: 64× 4B keys or 32× 8B records) into 8 YMM registers (AVX2) or 16 Q registers (ARM64 NEON).
   - Sorts columns vertically with **zero shuffles**, transposes the register matrix in place, and merges all vectors in registers into a sorted 256B quad-leaf—eliminating the first two intra-tile merge passes (`width = LEAF_SIZE, 2 * LEAF_SIZE`).
   - On ARM64 NEON 8B mode, hardware de-interleaving load **`vld2q_s32`** splits 32 records into **8 key registers (`k0..k7`) and 8 ID registers (`i0..i7`) (AoS → SoA)** so 8B records are sorted 4 lanes at a time just like 4B keys, then re-interleaved via **`vst2q_s32`**.
2. **Architecture-Specific Fast Bitonic Butterflies (AVX2 In-Lane Shuffles & NEON Symmetric 3-Stage `vzip` Pipeline)**:
   - On AVX2, `bitonic_step_8b` decides swaps for both lanes with a **single `_mm256_cmpgt_epi32`** while preserving equal-key records.
   - On ARM64 NEON, replaces `vtrn1q_s64`/`vtrn2q_s64` (which Clang compiled into extra `mov` + `ins v.d[1], v.d[0]` instructions) with a **symmetric 3-stage `vzip1q_s32`/`vzip2q_s32` butterfly** with zero register-move overhead.
3. **Dual-Chain (`carry_fwd` + `carry_bwd`) Branchless SIMD Merge with Off-Path Reversal & `__builtin_unpredictable`**:
   - Advances two independent SIMD bitonic merge chains from the front and back simultaneously, doubling ILP and breaking the 84M-cycle single-chain latency floor.
   - Reverses the freshly loaded vector (`low` in forward merge, `high` in backward merge) so neither `carry_fwd` nor `carry_bwd` has reversal latency on its loop-carried critical path.
   - Uses `HSORT_UNPREDICTABLE` (`__builtin_unpredictable`) to prevent LLVM AArch64 De-If-Conversion, guaranteeing 100% branchless `csel`/`cset` execution.
   - In 8B NEON merge (`merge_simd_neon`), uses 4-record SoA (`vld2q_s32` / `vst2q_s32`) to amortize the 6-cycle L1D load-to-address pointer dependency across 4 records (**1.5 cycles/record**).
4. **Adaptive Distribution Fast-Paths**:
   - **Pre-Sorted Quad-Leaf Clean Cache-Line Preservation** (`is_quad_leaf_sorted_avx2` / `is_quad_leaf_sorted_neon`).
   - **`O(1)` Disjoint Reversed-Run Fast-Path** (`if (src[end - 1] <= src[begin])`) in `pingpong_merge_pair`, cutting `6_Organ_Pipe` runtime by 2.4×.
   - **Zero-Allocation In-Place Bounded Insertion Sort** (`try_bounded_inplace_insertion_sort`) in `probe_and_handle_global_trend`, finishing `4_Mostly_Descending` in **0.78 ms (4B) / 0.86 ms (8B)** with zero heap allocations and zero page faults.
   - **Equal-Key Plateau Streamer** (`if (kl == kr && l_fwd[LANES - 1] == kl) [[unlikely]]`) to stream duplicate plateaus at L1 copy bandwidth.

---

## 4. ARM64 NEON Iterative Engineering Log (Baseline → v1 → v2 → v3 → v4)

Measured on **Snapdragon X Elite (`ARM64_Oryon`, `clang++ 23.1.5 -O3`, `N = 1,000,000`, median of 3 repeats)**:

| Iteration | Key Design Changes & Trial-and-Error Findings | 4B `Random` | 4B `Buckets` | 4B `Descending` | 4B `OrganPipe` | 8B `Random` | 8B `Descending` | 8B `OrganPipe` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline (Before)** | • Idea 6 ARM64 fell back to scalar Idea 4 for `N > 16`<br/>• Clang de-if-converted pointer updates to `b.ge`/`b.le` branches | 66.54 ms<br/>*(0.28×)* | 37.57 ms<br/>*(0.18×)* | 2.78 ms<br/>*(1.70×)* | 7.57 ms<br/>*(4.83×)* | 69.52 ms<br/>*(0.74×)* | 6.17 ms<br/>*(0.63×)* | 11.55 ms<br/>*(3.38×)* |
| **v1** | • Added `__builtin_unpredictable` (`csel`/`cset` enforced)<br/>• Added 4B/8B 256B Quad-Leaf & 4B Dual-Chain NEON merge<br/>• *Bottlenecks found*: Clang `vtrn1q_s64` emitted `mov+ins`; 8B scalar merge had 6cy/record address latency; `OrganPipe` overhead | 9.54 ms<br/>*(1.98×)* | 9.54 ms<br/>*(0.73×)* | 2.18 ms<br/>*(2.17×)* | 5.50 ms<br/>*(6.69×)* | 24.22 ms<br/>*(2.12×)* | 4.21 ms<br/>*(0.97×)* | 15.70 ms<br/>*(2.51×)* |
| **v2** | • Replaced `vtrn` with 3-stage symmetric `vzip1q`/`vzip2q` butterfly<br/>• Implemented 8B Dual-Chain 4-Record SoA NEON merge (`vld2q_s32`)<br/>• Added `O(1)` Disjoint Reversed-Run Fast-Path<br/>• *Bottleneck found*: 8 MB scratch page faults (\~3.5 ms) on `Mostly_Descending` | 9.61 ms<br/>*(1.97×)* | 9.88 ms<br/>*(0.71×)* | 1.95 ms<br/>*(2.43×)* | 2.98 ms<br/>*(12.28×)* | 18.31 ms<br/>*(2.81×)* | 4.34 ms<br/>*(0.92×)* | 6.56 ms<br/>*(5.95×)* |
| **v3 (Final Selected)** | • Added Zero-Allocation In-Place Bounded Insertion Sort<br/>• Added Equal-Key Plateau Streamer & single-load 8B sortedness check | **9.40 ms**<br/>**(2.01×)** | **8.50 ms**<br/>**(0.81×)** | **0.78 ms**<br/>**(6.03×)** | **2.86 ms**<br/>**(12.77×)** | **18.18 ms**<br/>**(2.83×)** | **0.86 ms**<br/>**(4.48×)** | **6.10 ms**<br/>**(6.39×)** |
| **v4 (Rejected)** | • *Hypothesis*: Change `trim_left` to `lower_bound` and add 2 scalar remainder loops inside hot SIMD loop<br/>• *Result*: Lost full first-bucket `memcpy` and increased loop I-cache pressure (`8.50 ms → 9.19 ms` on 4B `Buckets`); **reverted to v3** | 9.42 ms | 9.19 ms | 0.85 ms | 2.78 ms | 18.55 ms | 0.85 ms | 6.11 ms |

*(Ratios in parentheses denote speedup vs. `std::sort` on the same workload.)*

---

## 5. Head-to-Head Benchmark Results (Windows ARM64 NEON & Windows x64 AVX2)

### 5.1 Windows ARM64 (`Snapdragon X Elite / Qualcomm Oryon` · `clang++ 23.1.5 -O3` · `N = 1,000,000`)

- **Committed Raw Data**: [`docs/windowsARM/pure_cache_results_N=1000000.csv`](../windowsARM/pure_cache_results_N=1000000.csv) · [`docs/windowsARM/pure_cache_results_N=1000000.md`](../windowsARM/pure_cache_results_N=1000000.md)
- **Correctness**: Passed all **4,294 verification checks** in both `test_4b` and `test_8b`.

#### 4-Byte Integer Keys (`4B_Int32`, `N = 1,000,000`, median time in ms / hardware thread cycles in M)

| Workload | `std::sort` | `std::stable_sort` | Idea&nbsp;4 (Scalar) | Idea&nbsp;5 (NEON) | Idea&nbsp;6 (**Before**) | **Idea&nbsp;6 (After v3)** | vs.&nbsp;`std::sort` | vs.&nbsp;Before |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `0_Uniform_Random` | 18.86 (55.5M) | 13.55 (39.9M) | 62.50 (184.5M) | 28.82 (84.7M) | 66.54 (185.4M) | **9.40 (27.7M)** | **2.01×** | **7.08×** |
| `1_Frame_Jitter` | 16.30 (48.0M) | 11.94 (35.3M) | 30.91 (91.1M) | 15.96 (47.3M) | 31.22 (91.4M) | **6.21 (18.5M)** | **2.62×** | **5.03×** |
| `2_Append_5Pct_New` | 16.46 (48.4M) | 8.18 (24.2M) | 5.90 (17.4M) | 5.74 (16.8M) | 6.49 (17.5M) | **3.24 (9.6M)** | **5.07×** | **2.00×** |
| `3_Distance_Buckets` | 6.89 (20.2M) | 14.35 (42.3M) | 37.35 (110.0M) | 28.80 (84.8M) | 37.57 (110.7M) | **8.50 (24.7M)** | 0.81× *(1.69× vs stable)* | **4.42×** |
| `4_Mostly_Descending` | 4.72 (13.9M) | 11.55 (34.0M) | 3.33 (9.5M) | 3.55 (10.4M) | 2.78 (8.1M) | **0.78 (2.3M)** | **6.03×** | **3.55×** |
| `5_Gaussian_Normal` | 14.99 (44.2M) | 16.95 (50.1M) | 57.92 (170.8M) | 28.60 (84.5M) | 58.12 (170.7M) | **9.74 (28.8M)** | **1.54×** | **5.96×** |
| `6_Organ_Pipe` | 36.58 (108.0M) | 10.88 (32.0M) | 7.53 (22.2M) | 9.67 (28.5M) | 7.57 (22.2M) | **2.86 (8.4M)** | **12.77×** | **2.64×** |

#### 8-Byte Key+ID Records (`8B_Record`, `N = 1,000,000`, median time in ms / hardware thread cycles in M)

| Workload | `std::sort` | `std::stable_sort` | Idea&nbsp;4 (Scalar) | Idea&nbsp;5 (NEON) | Idea&nbsp;6 (**Before**) | **Idea&nbsp;6 (After v3)** | vs.&nbsp;`std::sort` | vs.&nbsp;Before |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `0_Uniform_Random` | 51.36 (150.8M) | 44.03 (129.7M) | 68.43 (201.9M) | 51.21 (151.0M) | 69.52 (204.6M) | **18.18 (54.0M)** | **2.83×** | **3.82×** |
| `1_Frame_Jitter` | 28.81 (85.3M) | 34.92 (103.1M) | 37.44 (110.3M) | 28.85 (85.5M) | 37.99 (111.7M) | **13.51 (39.7M)** | **2.13×** | **2.81×** |
| `2_Append_5Pct_New` | 18.10 (53.4M) | 19.88 (58.5M) | 9.08 (26.5M) | 10.89 (32.0M) | 9.66 (28.4M) | **6.30 (18.7M)** | **2.87×** | **1.53×** |
| `3_Distance_Buckets` | 20.80 (61.5M) | 43.99 (129.7M) | 42.74 (126.4M) | 50.77 (150.0M) | 44.51 (131.0M) | **15.47 (45.6M)** | **1.35×** | **2.88×** |
| `4_Mostly_Descending` | 3.86 (11.5M) | 30.60 (90.8M) | 5.79 (17.1M) | 5.98 (17.7M) | 6.17 (18.2M) | **0.86 (2.6M)** | **4.48×** | **7.15×** |
| `5_Gaussian_Normal` | 44.63 (131.7M) | 43.88 (129.5M) | 63.59 (187.9M) | 50.85 (150.0M) | 64.22 (189.2M) | **18.95 (55.7M)** | **2.36×** | **3.39×** |
| `6_Organ_Pipe` | 38.99 (115.3M) | 25.91 (76.6M) | 11.07 (32.7M) | 18.40 (54.4M) | 11.55 (34.0M) | **6.10 (18.0M)** | **6.39×** | **1.89×** |

---

### 5.2 Windows x64 (`Intel Core Ultra 7 270K Plus` · `g++ -O3` · `N = 1,000,000`)

- **Summary CSVs**: [`key_only_4b_summary.csv`](key_only_4b_summary.csv) · [`key_id_8b_summary.csv`](key_id_8b_summary.csv)

#### 4-Byte Integer Keys (`key_only_4b`, median time in ms)

| Workload | `std::sort` | Idea&nbsp;3 | Idea&nbsp;4 | Idea&nbsp;5 | **Idea&nbsp;6** | vs.&nbsp;`std::sort` | vs.&nbsp;Idea&nbsp;4 | vs.&nbsp;Idea&nbsp;5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_3d_distances` | 45.54 | 24.51 | 22.56 | 13.14 | **6.76** | **6.73×** | **3.33×** | **1.94×** |
| `next_frame_motion` | 26.24 | 15.62 | 16.64 | 10.32 | **5.90** | **4.45×** | **2.82×** | **1.75×** |
| `append_5pct_new` | 17.50 | 3.76 | 4.57 | 3.63 | **3.06** | **5.71×** | **1.49×** | **1.19×** |
| `distance_buckets` | 22.88 | 24.42 | 22.27 | 13.10 | **6.55** | **3.49×** | **3.40×** | **2.00×** |
| `mostly_descending` | 3.30 | 19.30 | 2.98 | 2.82 | **2.23** | **1.48×** | **1.34×** | **1.27×** |
| `gaussian_normal` | 40.68 | 24.29 | 22.79 | 13.33 | **6.98** | **5.83×** | **3.27×** | **1.91×** |
| `organ_pipe` | 39.97 | 11.37 | 9.24 | 4.48 | **3.86** | **10.34×** | **2.39×** | **1.16×** |

#### 8-Byte Key+ID Records (`key_id_8b`, median time in ms)

| Workload | `std::sort` | Idea&nbsp;3 | Idea&nbsp;4 | Idea&nbsp;5 | **Idea&nbsp;6** | vs.&nbsp;`std::sort` | vs.&nbsp;Idea&nbsp;4 | vs.&nbsp;Idea&nbsp;5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_3d_distances` | 43.85 | 32.25 | 29.32 | 32.54 | **18.19** | **2.41×** | **1.61×** | **1.79×** |
| `next_frame_motion` | 25.90 | 19.05 | 21.99 | 21.62 | **14.00** | **1.85×** | **1.57×** | **1.54×** |
| `append_5pct_new` | 17.46 | 4.74 | 8.33 | 9.18 | **6.56** | **2.66×** | **1.27×** | **1.40×** |
| `distance_buckets` | 22.70 | 32.63 | 29.56 | 33.21 | **18.30** | **1.24×** | **1.62×** | **1.81×** |
| `mostly_descending` | 4.33 | 22.67 | 6.01 | 6.82 | **4.47** | **0.97×** | **1.34×** | **1.53×** |
| `gaussian_normal` | 39.14 | 32.21 | 29.85 | 33.04 | **18.75** | **2.09×** | **1.59×** | **1.76×** |
| `organ_pipe` | 40.33 | 12.45 | 13.07 | 13.67 | **10.75** | **3.75×** | **1.22×** | **1.27×** |
