# Cache-Aware Hybrid Sort Study (64B Cache-Line Optimization)

**English** | [한국어](README.ko.md)

> **🤖 AI Collaboration Disclosure**
> 
> I am an undergraduate student currently learning algorithms and computer architecture. As a beginner who is not yet fluent in writing low-level C++ optimization code, **I actively utilized Generative AI (ChatGPT and Gemini) to implement the C++20 source code, refactor the structure, and build the benchmarking suite.**
> However, **all the core optimization ideas, algorithmic hypotheses, and the iterative problem-solving flow documented in this repository are entirely my own.**

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
*"Quick Sort has a worst-case time complexity of O(N^2) and shares the same O(N\log N) average complexity as Heap Sort or Merge Sort. Why is it so much faster in practice?"*

After learning that Quick Sort's practical speed comes from **Cache Locality**—which leads to higher CPU **Cache Hit** rates—I connected this hardware concept with the block-partitioning philosophy of **Timsort** and formulated my initial hypothesis:

1. **Splitting Down to the 64-Byte CPU Cache Line:**
   Since a standard CPU cache line is **64 bytes** (16 `int32_t` elements), what if we divide the array down to exact 64-byte cache-line blocks where further recursion gives no cache-hit advantage, sort each micro-block with minimal instructions inside the L1 cache, and then merge them?
2. **Improving Predictability in Partitioning & Merging:**
   Building on this, what if we predict data distribution using left/right summation during partitioning, or move two pointers in opposite directions across the merge boundary (`left` moving right-to-left $\leftarrow$, `right` moving left-to-right $\rightarrow$) to maximize CPU **Branch Prediction** accuracy?

This repository documents the journey of turning these raw ideas into working C++20 code and benchmarking them against `std::sort` and `std::stable_sort`.

## Key Hardware Performance Highlights (`perf_event_open`)

> Compared against `std::sort` on the same machine and workload ($N = 1,000,000$, `int32_t`).  
> **Note:** Values in `A / B` denote **`Ours / std::sort`**. **↓** = lower is better (`-n%`) · **↑** = higher is better (`+n%`, `+n%p`).

#### 1. `boundary_trimmed_merge`
| Scenario | Runtime ↓ | IPC ↑ | L1 Hit Rate ↑ | L1 Misses ↓ | Branch Hit Rate ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `0_Uniform_Random` | **21.32 ms**<br>(`-60.2%`) | **2.19** / 0.63<br>(`+248%`) | **95.11%**<br>(`-2.72%p`) | **2.02M** / 662K<br>(`+205%`) | **95.02%**<br>(`+19.48%p`) |
| `1_Frame_Jitter` | **13.43 ms**<br>(`-53.7%`) | **2.01** / 0.85<br>(`+136%`) | **99.47%**<br>(`+1.43%p`) | **128K** / 497K<br>(`-74.2%`) | **93.78%**<br>(`+10.99%p`) |
| `2_Append_5Pct_Tail` | **2.99 ms**<br>(`-84.2%`) | **2.37** / 1.66<br>(`+42.8%`) | **95.48%**<br>(`-2.42%p`) | **289K** / 655K<br>(`-55.9%`) | **98.22%**<br>(`+4.26%p`) |
| `5_Gaussian_Normal` | **21.35 ms**<br>(`-52.5%`) | **2.19** / 0.62<br>(`+253%`) | **95.08%**<br>(`-2.77%p`) | **2.03M** / 588K<br>(`+246%`) | **95.11%**<br>(`+19.22%p`) |
| `6_Organ_Pipe` | **8.86 ms**<br>(`-43.5%`) | **3.00** / 1.57<br>(`+91.1%`) | **95.20%**<br>(`-1.01%p`) | **1.15M** / 2.75M<br>(`-58.3%`) | **99.71%**<br>(`+6.73%p`) |

#### 2. `tiled_pingpong_merge`
| Scenario | Runtime ↓ | IPC ↑ | L1 Hit Rate ↑ | L1 Misses ↓ | Branch Hit Rate ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `0_Uniform_Random` | **20.40 ms**<br>(`-61.9%`) | **2.21** / 0.63<br>(`+251%`) | **96.93%**<br>(`-0.90%p`) | **1.26M** / 662K<br>(`+89.7%`) | **94.89%**<br>(`+19.35%p`) |
| `1_Frame_Jitter` | **15.30 ms**<br>(`-47.2%`) | **1.83** / 0.85<br>(`+115%`) | **94.66%**<br>(`-3.38%p`) | **1.26M** / 497K<br>(`+154%`) | **93.12%**<br>(`+10.33%p`) |
| `2_Append_5Pct_Tail` | **4.19 ms**<br>(`-77.8%`) | **2.37** / 1.66<br>(`+42.8%`) | **86.24%**<br>(`-11.66%p`) | **1.31M** / 655K<br>(`+101%`) | **98.57%**<br>(`+4.61%p`) |
| `5_Gaussian_Normal` | **20.37 ms**<br>(`-54.7%`) | **2.21** / 0.62<br>(`+256%`) | **96.93%**<br>(`-0.92%p`) | **1.26M** / 588K<br>(`+114%`) | **94.88%**<br>(`+18.99%p`) |
| `6_Organ_Pipe` | **8.49 ms**<br>(`-45.8%`) | **2.70** / 1.57<br>(`+72.0%`) | **94.28%**<br>(`-1.93%p`) | **1.26M** / 2.75M<br>(`-54.3%`) | **99.50%**<br>(`+6.52%p`) |

> - `0_Uniform_Random`: Uniformly distributed random integers with no initial order (high entropy).
> - `1_Frame_Jitter`: Pre-sorted array with small local random noise applied to every element (nearly sorted).
> - `2_Append_5Pct_Tail`: 95% sorted prefix followed by 5% unsorted random elements at the tail.
> - `5_Gaussian_Normal`: Bell-curve normal distribution concentrated around the mean.
> - `6_Organ_Pipe`: Symmetric mountain-shaped array (ascending first half, descending second half).

> [!NOTE]
> 🚧 This README is a work in progress and will be updated regularly.

## License

This project is licensed under the [MIT License](LICENSE).
