# Cache-Aware Hybrid Sort Study (64B Cache-Line Optimization)

**English** | [한국어](README.ko.md)

> **🤖 AI Collaboration**
>
> **Let me be 100% transparent about what AI did—and what I did:**
>
> * **Heavy AI Assistance for Code & Low-Level Tuning:**  
>   I am an undergraduate student still learning algorithms and computer architecture. Because I am not yet skilled enough to write low-level C++ optimizations on my own, **I heavily relied on Generative AI (ChatGPT and Gemini) to write the C++20 code, apply low-level hardware tricks (e.g., branchless `cmov`, cache-line alignment, L1 tiling), and build the benchmarking/profiling suite.**
> * **Independent Hypotheses:**  
>   I came up with the initial hypotheses myself while studying sorting algorithms, rather than taking them from an existing paper or sorting implementation.
---

## Feedback & Contributions

**PLEASE roast my code. Seriously. I'm begging you. 😭🙏**

I'm just a university student trying to figure things out, and I still have a lot to learn (and probably a lot of terrible code to fix).

If you find **anything** wrong, inefficient, ugly, confusing, or just plain stupid in my code, PLEASE let me know.

I don't care if you're a beginner or a veteran. **Any feedback is welcome. Literally ANYTHING.**

- Found a bug? Please tell me.
- Know a better approach? PLEASE tell me.
- Think my code is a mess? You're probably right. Tell me why. 💀

Open an Issue, start a Discussion, or submit a Pull Request. Whatever works for you!

**Every bit of criticism helps me grow as a developer. Please don't hold back. ❤️**

---

## 💡 Motivation & How the Idea Started

While studying sorting algorithms, a fundamental question crossed my mind:
*"Quick Sort has a worst-case time complexity of $O(N^2)$ and shares the same $O(N \log N)$ average complexity as Heap Sort or Merge Sort. Why can it be faster in practice?"*

Quicksort's practical performance depends on memory access patterns, comparison and movement costs, branch behavior, and implementation details. This led me to investigate small-block sorting and cache-aware merging.

1. **Splitting Down to the 64-Byte CPU Cache Line:**
   On the target CPU, a 64-byte cache line holds 16 int32_t elements. I wanted to test whether sorting micro-blocks of this size before merging could reduce overhead.
2. **Improving Predictability in Partitioning & Merging:**
   Could partition statistics improve pivot choices, and could opposite-direction boundary searches reduce unnecessary merge work? I also wanted to measure how these changes affected branch behavior.

This repository documents the journey of turning these raw ideas into working C++20 code and benchmarking them against `std::sort` and `std::stable_sort`.

## Key Hardware Performance Highlights (`perf_event_open`)

> Compared against `std::sort` on the same machine and workload ($N = 1,000,000$, `int32_t`).  
> **Note:** Values in `A / B` denote **`Ours / std::sort`**. **↓** = lower is better (`-n%`) · **↑** = higher is better (`+n%`, `+n%p`).

#### 1. `boundary_trimmed_merge`
| Scenario | Runtime ↓ | IPC ↑ | L1D Read Hit Rate ↑ | L1D Read Misses ↓ | Branch Hit Rate ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `0_Random_3D_Distances` | **21.32 ms**<br>(`-60.2%`) | **2.19** / 0.63<br>(`+248%`) | **95.11%**<br>(`-2.72%p`) | **2.02M** / 662K<br>(`+205%`) | **95.02%**<br>(`+19.48%p`) |
| `1_Frame_Jitter` | **13.43 ms**<br>(`-53.7%`) | **2.01** / 0.85<br>(`+136%`) | **99.47%**<br>(`+1.43%p`) | **128K** / 497K<br>(`-74.2%`) | **93.78%**<br>(`+10.99%p`) |
| `2_Append_5Pct_Tail` | **2.99 ms**<br>(`-84.2%`) | **2.37** / 1.66<br>(`+42.8%`) | **95.48%**<br>(`-2.42%p`) | **289K** / 655K<br>(`-55.9%`) | **98.22%**<br>(`+4.26%p`) |
| `5_Gaussian_Like` | **21.35 ms**<br>(`-52.5%`) | **2.19** / 0.62<br>(`+253%`) | **95.08%**<br>(`-2.77%p`) | **2.03M** / 588K<br>(`+246%`) | **95.11%**<br>(`+19.22%p`) |
| `6_Organ_Pipe` | **8.86 ms**<br>(`-43.5%`) | **3.00** / 1.57<br>(`+91.1%`) | **95.20%**<br>(`-1.01%p`) | **1.15M** / 2.75M<br>(`-58.3%`) | **99.71%**<br>(`+6.73%p`) |

#### 2. `tiled_pingpong_merge`
| Scenario | Runtime ↓ | IPC ↑ | L1D Read Hit Rate ↑ | L1D Read Misses ↓ | Branch Hit Rate ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `0_Random_3D_Distances` | **20.40 ms**<br>(`-61.9%`) | **2.21** / 0.63<br>(`+251%`) | **96.93%**<br>(`-0.90%p`) | **1.26M** / 662K<br>(`+89.7%`) | **94.89%**<br>(`+19.35%p`) |
| `1_Frame_Jitter` | **15.30 ms**<br>(`-47.2%`) | **1.83** / 0.85<br>(`+115%`) | **94.66%**<br>(`-3.38%p`) | **1.26M** / 497K<br>(`+154%`) | **93.12%**<br>(`+10.33%p`) |
| `2_Append_5Pct_Tail` | **4.19 ms**<br>(`-77.8%`) | **2.37** / 1.66<br>(`+42.8%`) | **86.24%**<br>(`-11.66%p`) | **1.31M** / 655K<br>(`+101%`) | **98.57%**<br>(`+4.61%p`) |
| `5_Gaussian_Like` | **20.37 ms**<br>(`-54.7%`) | **2.21** / 0.62<br>(`+256%`) | **96.93%**<br>(`-0.92%p`) | **1.26M** / 588K<br>(`+114%`) | **94.88%**<br>(`+18.99%p`) |
| `6_Organ_Pipe` | **8.49 ms**<br>(`-45.8%`) | **2.70** / 1.57<br>(`+72.0%`) | **94.28%**<br>(`-1.93%p`) | **1.26M** / 2.75M<br>(`-54.3%`) | **99.50%**<br>(`+6.52%p`) |

Lower runtime did not consistently coincide with higher L1D read hit rates or fewer L1D read misses. These measurements alone do not establish improved cache behavior as the cause of the speedup.
> - `0_Random_3D_Distances`: Squared-distance keys computed from uniformly sampled 3D coordinates. The resulting keys are not uniformly distributed.
> - `1_Frame_Jitter`: Pre-sorted array with small local random noise applied to every element (nearly sorted).
> - `2_Append_5Pct_Tail`: A 95% sorted prefix followed by 5% unsorted new distance keys.
> - `3_Distance_Buckets`: Squared-distance keys quantized into ~300 discrete buckets (`key /= 1,000,000`), creating heavy key duplication.
> - `4_Mostly_Descending`: Reverse-sorted array (descending order) perturbed by 1% random adjacent swaps.
> - `5_Gaussian_Like`: A bell-shaped distribution generated by summing four uniform integer samples.
> - `6_Organ_Pipe`: Symmetric mountain-shaped array (ascending first half, descending second half).

## 🧠 Thought Process & Evolution of Ideas
This project grew out of a series of "what if" experiments (`Idea 1` $\rightarrow$ `Idea 4`). While the core algorithmic concepts came from my own hypotheses, **I actively leaned on AI to translate those ideas into low-level C++ optimizations** (such as branchless `cmov` loops, galloping boundary search, and L1 cache tiling) that I wouldn't have known how to write on my own as an undergraduate.

### Idea 1: 64-Byte Cache-Line Cutoff Quick Sort (`cache_leaf_quick`)
* **My Idea:** I tested switching from Quicksort partitioning to insertion sort for partitions of at most 16 int32_t elements. This cutoff aims to reduce partitioning overhead; it does not guarantee cache-line alignment or L1 residency.
* **Outcome:** It was slightly faster than basic Quick Sort, but since the overall structure was still top-down Quick Sort, the improvement was pretty modest.
  
### Idea 2: Mean-Pivot Prediction via Summation (`mean_pivot_quick`)
* **My Idea:** The current implementation first performs Hoare partitioning, then scans the left and right partitions again to compute their sums and minimum/maximum keys. These statistics help choose subsequent pivots and identify all-equal ranges. This introduces an additional scan rather than accumulating statistics during partitioning.
* **Outcome:** It worked nicely on symmetric distributions and skipped all-duplicate ranges (`min == max`) quickly, but maintaining running sums added overhead and still couldn't solve Quick Sort's heavy branch mispredictions on random data.
  
### Idea 3: 64B Micro-Block + Cross-Pointer Boundary-Trimmed Merge (`boundary_trimmed_merge`)
* **My Idea:** Seeing the limits of top-down Quick Sort, I flipped the approach **bottom-up**—starting from Merge Sort instead:
  1. Instead of recursively splitting the array down to 1 element like standard Merge Sort, slice the array directly into **64-byte (16-element) blocks** and sort each block with Insertion Sort.
  2. When merging two adjacent sorted blocks, instead of blindly merging the whole range from start to end, start two pointers at the meeting boundary (`mid`) and walk them outward in opposite directions (`left` $\leftarrow$ `mid` $\rightarrow$ `right`) to **trim away outer elements that are already in order**, merging only the overlapping middle section.
* **AI-Assisted Low-Level Optimization:** AI helped implement bidirectional merge loops using conditional selection, exponential boundary searches, and half-buffer merging. These techniques aim to reduce data-dependent branching and memory copying. The generated instructions depend on the compiler and target, and loop and boundary branches remain.
* **Outcome:** Among the ideas I tried, **this was the one that finally showed a genuinely meaningful jump in the benchmarks**, especially on partially ordered or jittered data.
  
### Idea 4: 8 KiB L1 Tiling + Ping-Pong Merge (`tiled_pingpong_merge`)
* **Thought & AI Collaboration:** Both current merge variants use 8 KiB tiles. This variant additionally alternates source and destination buffers between merge passes to reduce copy-back operations. Ordered ranges still need to be copied to the destination buffer when required by the next pass.
* **Outcome:** On the random 3D-distance workload, this variant recorded slightly lower runtime and fewer L1D read misses than boundary_trimmed_merge.

### 💡 Takeaway from the Idea Evolution
Looking back at the progression from `Idea 1` $\rightarrow$ `Idea 4`, **`Idea 3` (`boundary_trimmed_merge`)**—along with `Idea 4` as its random-data variant—turned out to be the most meaningful outcome among my attempts.
Of course, much smarter people out there have already designed far better and faster sorting algorithms than this. Still, seeing this rough idea actually hold its own against `std::sort` in the benchmarks made me think: *"At least `Idea 3` seems decent enough—maybe it could actually be somewhat useful in the right situation?"*


> [!NOTE]
> 🚧 This README is a work in progress and will be updated regularly.

## License

This project is licensed under the [MIT License](LICENSE).
