#include "basalt/kernel/cpu_features.hpp"
#include "basalt/internal/simd_dispatch.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>

// MSVC has no __builtin_cpu_supports, so it queries CPUID/XGETBV itself. The
// same code can be forced on GCC/Clang with -DBASALT_CPUID_MANUAL (used to
// cross-check it against __builtin_cpu_supports).
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)) && !defined(BASALT_CPUID_MANUAL)
#define BASALT_CPUID_MANUAL 1
#endif

#if defined(BASALT_CPUID_MANUAL)
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace basalt::kernel {

namespace {

#if defined(BASALT_CPUID_MANUAL)

struct CpuidRegs {
    unsigned eax, ebx, ecx, edx;
};

CpuidRegs cpuid(unsigned leaf, unsigned subleaf) {
    CpuidRegs r{0, 0, 0, 0};
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<unsigned>(regs[0]);
    r.ebx = static_cast<unsigned>(regs[1]);
    r.ecx = static_cast<unsigned>(regs[2]);
    r.edx = static_cast<unsigned>(regs[3]);
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

unsigned long long xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}

CpuFeatures detect_hardware() {
    CpuFeatures f;
    const CpuidRegs l0 = cpuid(0, 0);
    if (l0.eax < 1) {
        return f;
    }
    const CpuidRegs l1 = cpuid(1, 0);
    const bool osxsave = ((l1.ecx >> 27) & 1u) != 0;
    const bool avx = ((l1.ecx >> 28) & 1u) != 0;
    const bool fma = ((l1.ecx >> 12) & 1u) != 0;
    if (!osxsave || !avx) {
        return f;  // No OS-managed AVX state: nothing wider than SSE is usable.
    }
    const unsigned long long xcr0 = xgetbv0();
    const bool ymm_ok = (xcr0 & 0x6u) == 0x6u;      // XMM and YMM state enabled
    const bool zmm_ok = (xcr0 & 0xE6u) == 0xE6u;    // + opmask, ZMM_Hi256, Hi16_ZMM
    if (!ymm_ok) {
        return f;
    }
    f.fma = fma;
    if (l0.eax >= 7) {
        const CpuidRegs l7 = cpuid(7, 0);
        f.avx2 = ((l7.ebx >> 5) & 1u) != 0;
        if (zmm_ok) {
            f.avx512f = ((l7.ebx >> 16) & 1u) != 0;
            f.avx512dq = ((l7.ebx >> 17) & 1u) != 0;
            f.avx512bw = ((l7.ebx >> 30) & 1u) != 0;
            f.avx512vl = ((l7.ebx >> 31) & 1u) != 0;
        }
    }
    return f;
}

#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))

CpuFeatures detect_hardware() {
    // __builtin_cpu_supports also verifies that the OS saves the YMM/ZMM state.
    __builtin_cpu_init();
    CpuFeatures f;
    f.avx2 = __builtin_cpu_supports("avx2");
    f.fma = __builtin_cpu_supports("fma");
    f.avx512f = __builtin_cpu_supports("avx512f");
    f.avx512dq = __builtin_cpu_supports("avx512dq");
    f.avx512bw = __builtin_cpu_supports("avx512bw");
    f.avx512vl = __builtin_cpu_supports("avx512vl");
    return f;
}

#else

CpuFeatures detect_hardware() {
    return CpuFeatures{};  // Unknown target: scalar only.
}

#endif

/// Parses BASALT_SIMD_MAX. Returns Auto (meaning "no cap") when unset/invalid.
SimdMode env_cap() {
    const char* raw = std::getenv("BASALT_SIMD_MAX");
    if (raw == nullptr) {
        return SimdMode::Auto;
    }
    std::string v(raw);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "scalar") return SimdMode::Scalar;
    if (v == "avx2") return SimdMode::AVX2;
    if (v == "avx512") return SimdMode::AVX512;
    return SimdMode::Auto;
}

CpuFeatures apply_cap(CpuFeatures f, SimdMode cap) {
    switch (cap) {
    case SimdMode::Scalar:
        return CpuFeatures{};
    case SimdMode::AVX2:
        f.avx512f = f.avx512dq = f.avx512bw = f.avx512vl = false;
        return f;
    case SimdMode::Auto:
    case SimdMode::AVX512:
        return f;
    }
    return f;
}

std::atomic<int> g_test_cap{static_cast<int>(SimdMode::Auto)};

detail::SimdLevel level_of(const CpuFeatures& f) {
    if (f.can_use_avx512()) return detail::SimdLevel::AVX512;
    if (f.can_use_avx2()) return detail::SimdLevel::AVX2;
    return detail::SimdLevel::Scalar;
}

detail::SimdLevel level_of_mode(SimdMode m) {
    switch (m) {
    case SimdMode::Scalar: return detail::SimdLevel::Scalar;
    case SimdMode::AVX2: return detail::SimdLevel::AVX2;
    case SimdMode::AVX512:
    case SimdMode::Auto: return detail::SimdLevel::AVX512;
    }
    return detail::SimdLevel::AVX512;
}

} // namespace

const CpuFeatures& cpu_features() {
    // Function-local static: initialised once, thread-safe (C++11 magic statics).
    static const CpuFeatures features = apply_cap(detect_hardware(), env_cap());
    return features;
}

namespace detail {

SimdLevel max_simd_level() {
    static const SimdLevel hardware = level_of(cpu_features());
    const int cap = g_test_cap.load(std::memory_order_relaxed);
    if (cap == static_cast<int>(SimdMode::Auto)) {
        return hardware;
    }
    return std::min(hardware, level_of_mode(static_cast<SimdMode>(cap)));
}

SimdLevel resolve_simd_level(SimdMode requested) {
    const SimdLevel best = max_simd_level();
    if (requested == SimdMode::Auto) {
        return best;
    }
    return std::min(best, level_of_mode(requested));
}

void set_simd_cap_for_testing(SimdMode cap) {
    g_test_cap.store(static_cast<int>(cap), std::memory_order_relaxed);
}

SimdMode simd_cap_for_testing() {
    return static_cast<SimdMode>(g_test_cap.load(std::memory_order_relaxed));
}

ScopedSimdCap::ScopedSimdCap(SimdMode cap) : m_previous(simd_cap_for_testing()) {
    set_simd_cap_for_testing(cap);
}

ScopedSimdCap::~ScopedSimdCap() {
    set_simd_cap_for_testing(m_previous);
}

} // namespace detail

} // namespace basalt::kernel
