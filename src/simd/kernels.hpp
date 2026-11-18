#pragma once
#include "hybrid_sorts.hpp"

namespace hsort::simd {

#ifdef BENCH_RECORD
static_assert(sizeof(Item) == 8 && offsetof(Item, key) == 0 && offsetof(Item, entity_id) == 4);
#endif

using Merge = decltype(&detail::bidirectional_branchless_merge);
struct Kernels {
    void (*full_leaf)(Item*) noexcept;
    Merge merge;
};

constexpr Kernels scalar{nullptr, detail::bidirectional_branchless_merge};
extern const Kernels sse2, avx2, avx512f, neon;

void tiled_sort(Item* data, std::size_t n, Kernels kernels);

} // namespace hsort::simd
