#pragma once

namespace basalt::kernel {

/// SIMD path preference for kernels that accept one (for example
/// apply_fk_filter). Every request is best-effort: if the CPU (or the
/// BASALT_SIMD_MAX cap, see cpu_features.hpp) does not support the requested
/// instruction set, the best supported lower path is used instead.
enum class SimdMode {
    Auto,    ///< Best path the CPU supports
    AVX2,    ///< AVX2 + FMA (never AVX-512); falls back to Scalar
    AVX512,  ///< AVX-512; falls back to AVX2, then Scalar
    Scalar   ///< Portable scalar code, no AVX instructions
};

} // namespace basalt::kernel
