# Idea 6: Transpose-Vectorized Quad-Leaf & Dual-Chain Branchless SIMD Tiled Merge Sort

[한국어](README.ko.md) | **English**

This document details the hypothesis, microarchitectural bottleneck analysis, five architectural mechanisms, and head-to-head benchmark results of **Idea 6 (`simd_transpose_merge` / `simd_transpose_tiled_merge_sort`)**, designed to resolve the bottlenecks observed in Idea 5 by lifting the core principles of Ideas 1, 3, and 4 directly into SIMD register architecture.

---

## 1. Motivation and Evolution from Ideas 1–5

1. **Idea 1 (64B Cache-Line Leaf Cutoff)**: Stopping partitioning at 64 bytes (`LEAF_SIZE`: 16× 4B keys or 8× 8B records) removes recursion and branch overhead on tiny subarrays.
2. **Idea 3 (Bidirectional Branchless Merge + `O(1)` Outer-Boundary Check)**:
   - Merging from the front (`->`, minimum selection) and back (`<-`, maximum selection) simultaneously splits the loop-carried dependency chain into two independent chains, doubling instruction-level parallelism (ILP).
   - Checking `data[begin] > right_min` and `data[end - 1] < left_max` in `O(1)` before binary searches avoids redundant boundary probing on random data.
3. **Idea 4 (Global Trend Probe + 8 KiB L1 Tiling + Ping-Pong Merge)**:
   - Completing leaf sorts and early merge passes inside 8 KiB L1 tiles (`TILE_SIZE`) while alternating source and destination buffers eliminates per-pass copy-back overhead.
4. **Idea 5 (Per-Leaf Horizontal Bitonic Network + Single-Chain Vector Merge)**:
   - Vectorized individual 64B leaves and overlap merges using horizontal variable permutes (`vpermd`) and a single-direction `carry` loop. While effective on 4B leaves, **8B records (`{ key, entity_id }`) regressed behind scalar `Idea 4`**, and single-chain vector merging suffered from serial `carry` latency.

---

## 2. Five Architectural Pillars of Idea 6

Implemented in [`src/idea6_simd_transpose_merge.cpp`](../../src/idea6_simd_transpose_merge.cpp):

1. **Vertical Sorting Network + In-Register Matrix Transpose for 256B Quad-Leaves (`sort_quad_leaf_avx2`)**:
   Loads four cache lines (`4 * LEAF_SIZE = 256B` = 8 YMM registers: 64× 4B keys or 32× 8B records), sorts columns vertically with **zero shuffles** (19-comparator 8-wire network in 4B; two parallel 5-comparator 4-wire networks in 8B), transposes the register matrix ($8 \times 8$ or $4 \times 4 \times 2$), and merges all 8 sorted vectors in registers into a single sorted 256B quad-leaf. This eliminates the first two intra-tile merge passes (`width = LEAF_SIZE` and `2 * LEAF_SIZE`) while keeping the remaining 5 intra-tile passes odd.
2. **1-Cycle In-Lane Immediate Shuffles & Single-Comparison Tie-Safe 8B Kernel (`bitonic_step_8b`)**:
   Replaces 3-cycle cross-lane variable permutes (`vpermd`) with 1-cycle in-lane `_mm256_shuffle_epi32` and immediate `_mm256_blend_epi32`. For 8B records, pre-aligning `l = blend(v, peer, Mask)` and `r = blend(peer, v, Mask)` decides swaps for both lanes with a **single `_mm256_cmpgt_epi32`** while preserving equal-key records. On ARM64 NEON, hardware `vld2q_s32`/`vst2q_s32` de-interleaves keys and IDs into separate registers for 4-lane leaf sorting paired with 64-bit `csel` bidirectional merging.
3. **Dual-Chain (Forward + Backward) Branchless SIMD Bitonic Merge (`merge_simd_avx2`)**:
   Advances two independent SIMD bitonic merge chains (`carry_fwd` from the front and `carry_bwd` from the back) in the same loop iteration with branchless `cmov` pointer updates, doubling SIMD merge ILP and eliminating data-dependent branch mispredictions.
4. **Clean Cache-Line Preservation, `O(1)` Outer-Boundary Check & Run-Boundary Short-Circuiting**:
   `is_quad_leaf_sorted_avx2` avoids memory stores on already-sorted 256B blocks so cache lines stay Clean on nearly-sorted inputs; `O(1)` outer-boundary checks skip unnecessary binary searches on random data; `all_run_boundaries_ordered` short-circuits completed merge passes.
5. **Stack-Resident L1 Scratch for `N <= TILE_SIZE` (8 KiB) & Single-TU Static Inlining**:
   Eliminates heap allocation for `N <= TILE_SIZE` and avoids Windows x64 ABI non-volatile `xmm6`–`xmm15` save/restore overhead across leaf and merge calls.

---

## 3. Verification and Head-to-Head Benchmark Results

- **Platform**: Intel Core Ultra 7 270K Plus · Windows x64 · `g++ -O3 -std=c++20`
- **Correctness**: Passed all **4,294 verification checks** in both `test_4b` and `test_8b`.
- **Summary CSVs**: [`key_only_4b_summary.csv`](key_only_4b_summary.csv) · [`key_id_8b_summary.csv`](key_id_8b_summary.csv)

### 3.1 4-Byte Integer Keys (`key_only_4b`, `N = 1,000,000`, median time in ms)

| Workload | `std::sort` | Idea&nbsp;3 | Idea&nbsp;4 | Idea&nbsp;5 | **Idea&nbsp;6** | vs.&nbsp;`std::sort` | vs.&nbsp;Idea&nbsp;4 | vs.&nbsp;Idea&nbsp;5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_3d_distances` | 45.54 | 24.51 | 22.56 | 13.14 | **6.76** | **6.73×** | **3.33×** | **1.94×** |
| `next_frame_motion` | 26.24 | 15.62 | 16.64 | 10.32 | **5.90** | **4.45×** | **2.82×** | **1.75×** |
| `append_5pct_new` | 17.50 | 3.76 | 4.57 | 3.63 | **3.06** | **5.71×** | **1.49×** | **1.19×** |
| `distance_buckets` | 22.88 | 24.42 | 22.27 | 13.10 | **6.55** | **3.49×** | **3.40×** | **2.00×** |
| `mostly_descending` | 3.30 | 19.30 | 2.98 | 2.82 | **2.23** | **1.48×** | **1.34×** | **1.27×** |
| `gaussian_normal` | 40.68 | 24.29 | 22.79 | 13.33 | **6.98** | **5.83×** | **3.27×** | **1.91×** |
| `organ_pipe` | 39.97 | 11.37 | 9.24 | 4.48 | **3.86** | **10.34×** | **2.39×** | **1.16×** |

### 3.2 8-Byte Key+ID Records (`key_id_8b`, `N = 1,000,000`, median time in ms)

| Workload | `std::sort` | Idea&nbsp;3 | Idea&nbsp;4 | Idea&nbsp;5 | **Idea&nbsp;6** | vs.&nbsp;`std::sort` | vs.&nbsp;Idea&nbsp;4 | vs.&nbsp;Idea&nbsp;5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_3d_distances` | 43.85 | 32.25 | 29.32 | 32.54 | **18.19** | **2.41×** | **1.61×** | **1.79×** |
| `next_frame_motion` | 25.90 | 19.05 | 21.99 | 21.62 | **14.00** | **1.85×** | **1.57×** | **1.54×** |
| `append_5pct_new` | 17.46 | 4.74 | 8.33 | 9.18 | **6.56** | **2.66×** | **1.27×** | **1.40×** |
| `distance_buckets` | 22.70 | 32.63 | 29.56 | 33.21 | **18.30** | **1.24×** | **1.62×** | **1.81×** |
| `mostly_descending` | 4.33 | 22.67 | 6.01 | 6.82 | **4.47** | **0.97×** | **1.34×** | **1.53×** |
| `gaussian_normal` | 39.14 | 32.21 | 29.85 | 33.04 | **18.75** | **2.09×** | **1.59×** | **1.76×** |
| `organ_pipe` | 40.33 | 12.45 | 13.07 | 13.67 | **10.75** | **3.75×** | **1.22×** | **1.27×** |
