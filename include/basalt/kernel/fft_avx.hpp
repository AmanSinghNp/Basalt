#pragma once

#include <cstddef>

namespace basalt::kernel {

/// AVX2-accelerated in-place forward FFT (Cooley-Tukey radix-2).
/// Uses FMA intrinsics for butterfly operations.
/// Falls back to scalar for n < 16.
///
/// CPU dispatch: the AVX2 code runs only when cpu_features() reports AVX2 and
/// FMA (and BASALT_SIMD_MAX does not cap it below avx2). Otherwise this call
/// transparently runs the portable scalar kernel (same results as fft_forward),
/// so it never executes an unsupported instruction and never throws because
/// of the CPU.
/// 'real' and 'imag' are separate SoA arrays of length 'n' (they must not alias).
/// Same sign convention and size rules as fft_forward (see fft.hpp):
/// n == 0 is a no-op, n == 1 is the identity, and any n that is not a power of
/// two (or exceeds 2^30) throws std::invalid_argument.
void fft_forward_avx(float* real, float* imag, size_t n);

/// AVX2-accelerated in-place inverse FFT with 1/N scaling.
/// Throws std::invalid_argument for n not a power of two.
void fft_inverse_avx(float* real, float* imag, size_t n);

} // namespace basalt::kernel
