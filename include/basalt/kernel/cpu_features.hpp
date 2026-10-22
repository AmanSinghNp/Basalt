#pragma once

// Runtime CPU feature detection and the compile-time target macros used to keep
// the library baseline at plain x86-64 (SSE2).
//
// Basalt is compiled WITHOUT -mavx2/-mfma. Every function that uses AVX2/FMA
// (or AVX-512) intrinsics carries a per-function target attribute and is only
// ever called after a runtime check, so the library loads and runs on any
// x86-64 CPU and uses the wide paths when the CPU (and OS) support them.

#include <cstddef>

// ---------------------------------------------------------------------------
// Target attribute macros
// ---------------------------------------------------------------------------
// BASALT_TARGET_AVX2   : put on every function that uses AVX/AVX2/FMA intrinsics.
// BASALT_TARGET_AVX512 : same for AVX-512F/DQ code (implies AVX2 and FMA).
//
// On GCC/Clang these expand to __attribute__((target(...))). On MSVC they are
// empty (MSVC accepts intrinsics without /arch flags). Note that GCC cannot
// inline a target("avx2") function into a function without that target, so any
// caller of an annotated inline helper (for example the fast_math.hpp
// functions) must itself carry the attribute and be guarded by a runtime
// cpu_features() check.
#if defined(__GNUC__) || defined(__clang__)
#define BASALT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define BASALT_TARGET_AVX512 __attribute__((target("avx2,fma,avx512f,avx512dq")))
#else
#define BASALT_TARGET_AVX2
#define BASALT_TARGET_AVX512
#endif

namespace basalt::kernel {

/// SIMD instruction sets usable on this machine by this process.
struct CpuFeatures {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512dq = false;
    bool avx512bw = false;
    bool avx512vl = false;

    /// True when the AVX2 kernels may run (they need both AVX2 and FMA).
    [[nodiscard]] bool can_use_avx2() const noexcept { return avx2 && fma; }

    /// True when the AVX-512 f-k filter may run. It needs AVX-512F and DQ on
    /// top of AVX2/FMA (it does not use the VL/BW subsets).
    [[nodiscard]] bool can_use_avx512() const noexcept {
        return can_use_avx2() && avx512f && avx512dq;
    }
};

/// Features of the running CPU, as usable by this process.
///
/// The result is computed once on first use (thread-safe) and then cached.
/// A feature is reported only if the CPU implements it AND the operating
/// system has enabled the matching register state (XSAVE/XCR0 YMM and ZMM
/// bits): GCC/Clang use __builtin_cpu_supports, which performs that check;
/// MSVC builds query CPUID and XGETBV directly.
///
/// Environment override: BASALT_SIMD_MAX = scalar | avx2 | avx512
/// (case-insensitive, read once together with the detection) caps the
/// reported features, for testing the fallback paths or for users who want to
/// pin a path:
///   scalar  : every feature is reported as absent (portable scalar kernels)
///   avx2    : AVX-512 features are reported as absent
///   avx512  : no cap
/// An unset, empty or unrecognised value means no cap. The cap can only
/// remove features, never add ones the CPU lacks.
///
/// Kernel dispatch uses these features plus (in tests only) an additional cap,
/// so cpu_features() always describes the hardware capability and the
/// BASALT_SIMD_MAX request, not an in-process test override.
const CpuFeatures& cpu_features();

} // namespace basalt::kernel
