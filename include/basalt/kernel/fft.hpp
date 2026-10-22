#pragma once

#include <cstddef>

namespace basalt::kernel {

/// Sign convention
/// ---------------
/// The forward transform is the engineering-convention DFT
///     X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N)
/// so a signal exp(+2*pi*i*k0*n/N) puts all of its energy in bin k0.
/// The inverse uses exp(+2*pi*i*k*n/N) and scales by 1/N.

/// In-place forward FFT (Decimation-in-Time, Cooley-Tukey radix-2).
/// 'real' and 'imag' are separate SoA arrays of length 'n' (they must not alias).
///
/// Size rules (checked in every build type):
///   - n == 0 is an empty transform and a no-op (pointers are not inspected).
///   - n == 1 is the identity.
///   - n must otherwise be a power of two no greater than 2^30; any other value
///     throws std::invalid_argument before any memory is touched.
///   - Null or aliasing data pointers (for n > 0) throw std::invalid_argument.
///
/// Twiddle factors and bit-reversal tables are cached per thread (bounded:
/// one plan per power of two up to 2^20; larger sizes are not cached), so the
/// function is safe to call concurrently on different data from many threads.
void fft_forward(float* real, float* imag, size_t n);

/// In-place inverse FFT with 1/N scaling. Same size and pointer rules as
/// fft_forward (throws std::invalid_argument for n not a power of two).
void fft_inverse(float* real, float* imag, size_t n);

} // namespace basalt::kernel
