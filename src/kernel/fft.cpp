#include "basalt/kernel/fft.hpp"

#include <cassert>
#include <cmath>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace basalt::kernel {

namespace {

/// Reverse the bottom log2(n) bits of 'x'.
size_t bit_reverse(size_t x, size_t log2n) {
    size_t result = 0;
    for (size_t i = 0; i < log2n; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/// Compute log2(n). Assumes n is a power of 2.
size_t log2_of(size_t n) {
    size_t result = 0;
    while ((1u << result) < n) {
        ++result;
    }
    return result;
}

/// Core in-place FFT. 'inverse' controls direction of twiddle rotation.
void fft_impl(float* real, float* imag, size_t n, bool inverse) {
    assert(n > 0 && (n & (n - 1)) == 0 && "n must be a power of 2");

    size_t log2n = log2_of(n);

    // --- Step 1: Bit-reversal permutation ---
    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, log2n);
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // --- Step 2: Butterfly stages ---
    // For each stage s (block size doubles: 2, 4, 8, ... n)
    for (size_t s = 1; s <= log2n; ++s) {
        size_t m = 1u << s;         // Block size at this stage
        size_t half_m = m >> 1;     // Half-block

        // Twiddle factor angle step
        // Forward: W = e^(-2πi/m), Inverse: W = e^(+2πi/m)
        double angle_step = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(m);

        // Process each block
        for (size_t k = 0; k < n; k += m) {
            // Compute twiddle factors for this block
            for (size_t j = 0; j < half_m; ++j) {
                double angle = angle_step * static_cast<double>(j);
                float wr = static_cast<float>(std::cos(angle));
                float wi = static_cast<float>(std::sin(angle));

                // Butterfly indices
                size_t even = k + j;
                size_t odd  = k + j + half_m;

                // t = W * X[odd]
                float tr = wr * real[odd] - wi * imag[odd];
                float ti = wr * imag[odd] + wi * real[odd];

                // Butterfly
                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] = real[even] + tr;
                imag[even] = imag[even] + ti;
            }
        }
    }

    // --- Step 3: Scale by 1/N for inverse ---
    if (inverse) {
        float scale = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }
}

} // anonymous namespace

void fft_forward(float* real, float* imag, size_t n) {
    fft_impl(real, imag, n, false);
}

void fft_inverse(float* real, float* imag, size_t n) {
    fft_impl(real, imag, n, true);
}

} // namespace basalt::kernel
