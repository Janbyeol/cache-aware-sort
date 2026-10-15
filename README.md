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
*"Quick Sort has a worst-case time complexity of $O(N^2)$ and shares the same $O(N \log N)$ average complexity as Heap Sort or Merge Sort. Why is it so much faster in practice?"*

After learning that Quick Sort's practical speed comes from **Cache Locality**—which leads to higher CPU **Cache Hit** rates—I connected this hardware concept with the block-partitioning philosophy of **Timsort** and formulated my initial hypothesis:

1. **Splitting Down to the 64-Byte CPU Cache Line:**
   Since a standard CPU cache line is **64 bytes** (16 `int32_t` elements), what if we divide the array down to exact 64-byte cache-line blocks where further recursion gives no cache-hit advantage, sort each micro-block with minimal instructions inside the L1 cache, and then merge them?
2. **Improving Predictability in Partitioning & Merging:**
   Building on this, what if we predict data distribution using left/right summation during partitioning, or move two pointers in opposite directions across the merge boundary (`left` moving right-to-left $\leftarrow$, `right` moving left-to-right $\rightarrow$) to maximize CPU **Branch Prediction** accuracy?

This repository documents the journey of turning these raw ideas into working C++20 code and benchmarking them against `std::sort` and `std::stable_sort`.


> [!NOTE]
> 🚧 This README is a work in progress and will be updated regularly.

## License

This project is licensed under the [MIT License](LICENSE).
