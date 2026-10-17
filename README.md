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
> **Note:** **Runtime** = lower is better (`-n%` is faster) · **Hit Rate** = higher is better (`+n%p` is more hits).

| Scenario | Algorithm | Runtime | L1 Hit Rate  | Branch Hit Rate  |
| :--- | :--- | ---: | ---: | ---: |
| `1_Frame_Jitter` | `boundary_trimmed_merge` | **13.43 ms** (`-53.7%`) | **99.47%** (`+1.43%p`) | **93.78%** (`+10.99%p`) |
| `2_Append_5Pct_Tail` | `boundary_trimmed_merge` | **2.99 ms** (`-84.2%`) | **95.48%** (`-2.42%p`) | **98.22%** (`+4.26%p`) |
| `6_Organ_Pipe` | `boundary_trimmed_merge` | **8.86 ms** (`-43.5%`) | **95.20%** (`-1.01%p`) | **99.71%** (`+6.73%p`) |
| `0_Uniform_Random` | `boundary_trimmed_merge` | **21.32 ms** (`-60.2%`) | **95.11%** (`-2.72%p`) | **95.02%** (`+19.48%p`) |
| `0_Uniform_Random` | `tiled_pingpong_merge` | **20.40 ms** (`-61.9%`) | **96.93%** (`-0.90%p`) | **94.89%** (`+19.35%p`) |
| `5_Gaussian_Normal` | `tiled_pingpong_merge` | **20.37 ms** (`-54.7%`) | **96.93%** (`-0.92%p`) | **94.88%** (`+18.99%p`) |

> - **`0_Uniform_Random`**: Uniformly distributed random integers with no initial order (high entropy).
> - **`1_Frame_Jitter`**: Pre-sorted array with small local random noise applied to every element (nearly sorted).
> - **`2_Append_5Pct_Tail`**: 95% sorted prefix followed by 5% unsorted random elements at the tail.
> - **`5_Gaussian_Normal`**: Bell-curve normal distribution concentrated around the mean.
> - **`6_Organ_Pipe`**: Symmetric mountain-shaped array (ascending first half, descending second half).

> [!NOTE]
> 🚧 This README is a work in progress and will be updated regularly.

## License

This project is licensed under the [MIT License](LICENSE).
