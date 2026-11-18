#include "simd/kernels.hpp"
#include <cstdlib>
#include <string_view>

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
#define HSORT_X64 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(_M_ARM64) || (defined(__aarch64__) && defined(__ARM_NEON))
#define HSORT_NEON 1
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

namespace hsort {
namespace {

#if defined(HSORT_X64)
struct CpuId { unsigned eax, ebx, ecx, edx; };

CpuId cpuid(unsigned leaf, unsigned subleaf = 0) noexcept {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    return {unsigned(regs[0]), unsigned(regs[1]), unsigned(regs[2]), unsigned(regs[3])};
#else
    CpuId regs;
    __cpuid_count(leaf, subleaf, regs.eax, regs.ebx, regs.ecx, regs.edx);
    return regs;
#endif
}

#if !defined(_MSC_VER)
__attribute__((target("xsave")))
#endif
unsigned long long vector_state() noexcept { return _xgetbv(0); }
#endif

struct Backend {
    std::string_view name;
    simd::Kernels kernels;
    bool supported;
};

struct Selection {
    simd::Kernels kernels;
    std::string label;
};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996) // getenv reads process settings once, during dispatch initialization.
#endif
std::string_view setting(const char* name, const char* fallback) noexcept {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

Selection select_kernels() {
    const Backend scalar{"scalar", simd::scalar, true};
    std::string_view cpu = "scalar";
#if defined(HSORT_X64)
    const unsigned max_leaf = cpuid(0).eax;
    constexpr unsigned osxsave_avx = (1u << 27) | (1u << 28);
    const auto state = (cpuid(1).ecx & osxsave_avx) == osxsave_avx ? vector_state() : 0;
    const auto features = max_leaf >= 7 ? cpuid(7) : CpuId{};
    const bool avx2 = (state & 0x6) == 0x6 && (features.ebx & (1u << 5));
    const bool avx512f = (state & 0xe6) == 0xe6 && (features.ebx & (1u << 16));
    const bool avx10_2 = max_leaf >= 0x24 && features.eax >= 1 &&
                         (cpuid(7, 1).edx & (1u << 19)) && (cpuid(0x24).ebx & 0xff) >= 2;
    cpu = avx10_2 ? "avx10.2" : avx512f ? "avx512f" : avx2 ? "avx2" : "sse2";
    // Untuned default. Wider kernels can be selected independently for comparison.
    const Backend choices[] = {{"avx2", simd::avx2, avx2}, {"sse2", simd::sse2, true},
                               {"avx512f", simd::avx512f, avx512f}, scalar};
#elif defined(HSORT_NEON)
#if defined(__linux__)
    const bool has_neon = (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#else
    constexpr bool has_neon = true; // Windows and Apple ARM64 ABIs include Advanced SIMD.
#endif
    if (has_neon) cpu = "neon";
    const Backend choices[] = {{"neon", simd::neon, has_neon}, scalar};
#else
    const Backend choices[] = {scalar};
#endif
    const auto isa = setting("HSORT_SIMD_ISA", "auto");
    const auto stage = setting("HSORT_SIMD_STAGE", "both");
    if (stage != "both" && stage != "leaf" && stage != "merge")
        throw std::invalid_argument("HSORT_SIMD_STAGE accepts both, leaf or merge");
    for (const Backend& backend : choices) {
        if (!backend.supported || (isa != "auto" && isa != backend.name)) continue;
        auto kernels = backend.kernels;
        if (stage == "leaf") kernels.merge = simd::scalar.merge;
        if (stage == "merge") kernels.full_leaf = nullptr;
        const auto active = backend.name == "scalar" ? std::string_view("scalar") : stage;
        std::string label = "simd_tiled_pingpong_merge.";
        label.append(backend.name).append(".").append(active).append(".cpu-").append(cpu);
        return {kernels, std::move(label)};
    }
    throw std::invalid_argument("Unavailable HSORT_SIMD_ISA '" + std::string(isa) +
                                "' on " + std::string(cpu));
}

const Selection& selection() {
    static const Selection selected = select_kernels();
    return selected;
}

} // namespace

const char* simd_sort_label() { return selection().label.c_str(); }

void simd_tiled_pingpong_merge_sort(Item* data, std::size_t n) {
    simd::tiled_sort(data, n, selection().kernels);
}

} // namespace hsort
