#pragma once

#include <cstddef>

namespace basalt::kernel {

/// In-place forward FFT (Decimation-in-Time, Cooley-Tukey radix-2).
/// 'real' and 'imag' are separate SoA arrays of length 'n'.
/// 'n' MUST be a power of 2.
void fft_forward(float* real, float* imag, size_t n);

/// In-place inverse FFT with 1/N scaling.
/// 'n' MUST be a power of 2.
void fft_inverse(float* real, float* imag, size_t n);

} // namespace basalt::kernel
