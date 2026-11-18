#pragma once

// Include inside each ISA-specific translation unit, under its instruction target.
namespace hsort::simd {
namespace {

template<class Ops>
struct Network {
    using Vec = typename Ops::Vec;
    static constexpr int words_per_item = sizeof(Item) / sizeof(std::int32_t);
    static constexpr int lanes = sizeof(Vec) / sizeof(Item);

    static Vec load(const Item* src) noexcept {
        Vec value;
        std::memcpy(&value, src, sizeof(Vec));
        return value;
    }

    static void store(Item* dst, Vec value) noexcept {
        std::memcpy(static_cast<void*>(dst), &value, sizeof(Vec));
    }

    // Each lane keeps its first operand on a tie. For two different registers,
    // reversing the operands for the maximum preserves both equal-key records.
    template<bool High>
    static Vec extreme(Vec a, Vec b) noexcept {
        if constexpr (words_per_item == 1) {
            if constexpr (High) return Ops::max32(a, b);
            else return Ops::min32(a, b);
        } else {
            const auto swap = High ? Ops::greater_key(b, a) : Ops::greater_key(a, b);
            return Ops::select(swap, b, a);
        }
    }

    template<int Span, int Distance>
    static constexpr unsigned high_words() noexcept {
        unsigned mask = 0;
        for (int word = 0; word < lanes * words_per_item; ++word) {
            const int lane = word / words_per_item;
            if (((lane & Span) != 0) != ((lane & Distance) != 0)) mask |= 1u << word;
        }
        return mask;
    }

    template<int Span, int Distance>
    static Vec merge_steps(Vec value) noexcept {
        const Vec peer = Ops::template permute_xor<Distance * words_per_item>(value);
        value = Ops::template blend<high_words<Span, Distance>()>(
            extreme<false>(value, peer), extreme<true>(value, peer));
        if constexpr (Distance > 1) return merge_steps<Span, Distance / 2>(value);
        else return value;
    }

    template<int Span = 2>
    static Vec sort_vector(Vec value) noexcept {
        value = merge_steps<Span, Span / 2>(value);
        if constexpr (Span < lanes) return sort_vector<Span * 2>(value);
        else return value;
    }

    // Merge two ascending vectors into sorted lower and upper halves.
    static void merge_vectors(Vec& lower, Vec& upper) noexcept {
        const Vec reversed = Ops::template permute_xor<(lanes - 1) * words_per_item>(upper);
        const Vec high = extreme<true>(reversed, lower);
        lower = merge_steps<lanes, lanes / 2>(extreme<false>(lower, reversed));
        upper = merge_steps<lanes, lanes / 2>(high);
    }

    static void full_leaf(Item* data) noexcept {
        constexpr int count = LEAF_SIZE / lanes;
        static_assert(count == 1 || count == 2 || count == 4);
        Vec block[count];
        for (int i = 0; i < count; ++i) block[i] = sort_vector(load(data + i * lanes));
        if constexpr (count >= 2) merge_vectors(block[0], block[1]);
        if constexpr (count == 4) {
            merge_vectors(block[2], block[3]);
            merge_vectors(block[0], block[2]);
            merge_vectors(block[1], block[3]);
            merge_vectors(block[1], block[2]);
        }
        for (int i = 0; i < count; ++i) store(data + i * lanes, block[i]);
    }

    static void merge(const Item* src, Item* dst, std::size_t begin,
                      std::size_t mid, std::size_t end) noexcept {
        if (mid - begin < lanes || end - mid < lanes) {
            detail::bidirectional_branchless_merge(src, dst, begin, mid, end);
            return;
        }

        const Item* left = src + begin;
        const Item* right = src + mid;
        const Item* left_end = right;
        const Item* right_end = src + end;
        Item* out = dst + begin;
        Vec low = load(left);
        Vec carry = load(right);
        left += lanes;
        right += lanes;
        merge_vectors(low, carry);
        store(out, low);
        out += lanes;

        // The next output vector must include the input with the smaller head.
        // If its block is incomplete, finish with that short run and the carry.
        while (left != left_end && right != right_end) {
            const bool take_left = *left <= *right;
            const Item*& next = take_left ? left : right;
            if ((take_left ? left_end : right_end) - next < lanes) break;
            low = load(next);
            next += lanes;
            merge_vectors(low, carry);
            store(out, low);
            out += lanes;
        }

        // At least one tail has fewer than 'lanes' items. Fold it into the
        // carry before merging with the other tail, which can still be long.
        if (left_end - left >= lanes) {
            std::swap(left, right);
            std::swap(left_end, right_end);
        }
        Item held[lanes], pending[2 * lanes];
        store(held, carry);
        Item* pending_end = std::merge(held, held + lanes, left, left_end, pending);
        std::merge(pending, pending_end, right, right_end, out);
    }
};

} // namespace
} // namespace hsort::simd
