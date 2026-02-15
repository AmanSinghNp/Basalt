#pragma once

#include <cstddef>

namespace basalt::kernel {

/// AVX2-accelerated in-place forward FFT (Cooley-Tukey radix-2).
/// Uses FMA intrinsics for butterfly operations.
/// Falls back to scalar for n < 16.
/// 'real' and 'imag' are separate SoA arrays of length 'n'.
/// 'n' MUST be a power of 2.
void fft_forward_avx(float* real, float* imag, size_t n);

/// AVX2-accelerated in-place inverse FFT with 1/N scaling.
/// 'n' MUST be a power of 2.
void fft_inverse_avx(float* real, float* imag, size_t n);

} // namespace basalt::kernel
