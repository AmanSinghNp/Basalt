#pragma once

// Internal SIMD dispatch helpers. Not part of the installed public API.
//
// Every AVX2/AVX-512 entry point asks resolve_simd_level() ONCE per call (never
// per row or per element) and then runs the matching implementation.

#include "basalt/kernel/cpu_features.hpp"
#include "basalt/kernel/simd_mode.hpp"

namespace basalt::kernel::detail {

enum class SimdLevel { Scalar = 0, AVX2 = 1, AVX512 = 2 };

/// Highest level the process may use: min(what cpu_features() reports, the
/// test cap installed by set_simd_cap_for_testing, if any).
SimdLevel max_simd_level();

/// Level to run for a caller request: Auto means max_simd_level(); an explicit
/// mode is clamped down to max_simd_level() (never up).
SimdLevel resolve_simd_level(SimdMode requested);

/// True when the AVX2+FMA kernels may run (shorthand for the common check).
inline bool use_avx2() { return max_simd_level() >= SimdLevel::AVX2; }

/// Test hook: caps the dispatch level for the whole process, on top of the
/// BASALT_SIMD_MAX environment variable. SimdMode::Auto removes the cap.
/// Thread-safe, but intended to be called between kernel calls, not during.
void set_simd_cap_for_testing(SimdMode cap);

/// Current test cap (Auto when none is installed).
SimdMode simd_cap_for_testing();

/// RAII helper: installs a cap and restores the previous one on destruction.
class ScopedSimdCap {
public:
    explicit ScopedSimdCap(SimdMode cap);
    ~ScopedSimdCap();
    ScopedSimdCap(const ScopedSimdCap&) = delete;
    ScopedSimdCap& operator=(const ScopedSimdCap&) = delete;

private:
    SimdMode m_previous;
};

} // namespace basalt::kernel::detail
