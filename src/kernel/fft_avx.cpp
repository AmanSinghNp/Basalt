#include "basalt/kernel/fft_avx.hpp"
#include "basalt/internal/fft_plan.hpp"

#include <immintrin.h>
#include <utility>

namespace basalt::internal {

namespace {

/// Multiply n elements of real/imag by 'scale' (AVX2).
void scale_avx(float* real, float* imag, size_t n, float scale_val) {
    const __m256 scale = _mm256_set1_ps(scale_val);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 r = _mm256_loadu_ps(real + i);
        __m256 im = _mm256_loadu_ps(imag + i);
        _mm256_storeu_ps(real + i, _mm256_mul_ps(r, scale));
        _mm256_storeu_ps(imag + i, _mm256_mul_ps(im, scale));
    }
    for (; i < n; ++i) {
        real[i] *= scale_val;
        imag[i] *= scale_val;
    }
}

/// Unscaled AVX2/FMA transform for n >= 16. 'Inverse' conjugates the twiddles.
template <bool Inverse>
void fft_avx_core(float* real, float* imag, const FFTPlan& plan) {
    const size_t n = plan.n;

    // --- Step 1: Bit-reversal permutation via the precomputed swap list ---
    {
        const uint32_t* sw = plan.swaps.data();
        const size_t num_swaps = plan.swaps.size() / 2;
        for (size_t p = 0; p < num_swaps; ++p) {
            const size_t i = sw[2 * p];
            const size_t j = sw[2 * p + 1];
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // --- Step 2: Butterfly stages ---
    for (size_t s = 1; s <= plan.log2n; ++s) {
        const size_t m = size_t{1} << s;
        const size_t half_m = m >> 1;
        const float* tw_real = plan.tw_real.data() + (half_m - 1);
        const float* tw_imag = plan.tw_imag.data() + (half_m - 1);

        for (size_t k = 0; k < n; k += m) {
            size_t j = 0;

            // AVX2 path: 8 butterflies at once (needs half_m >= 8).
            // The working set of a stage streams sequentially, so the hardware
            // prefetcher covers it; explicit software prefetch was removed
            // (it only ever reached 32 bytes ahead).
            for (; j + 7 < half_m; j += 8) {
                const size_t even_idx = k + j;
                const size_t odd_idx = even_idx + half_m;

                const __m256 wr = _mm256_loadu_ps(tw_real + j);
                const __m256 wi = _mm256_loadu_ps(tw_imag + j);

                const __m256 er = _mm256_loadu_ps(real + even_idx);
                const __m256 ei = _mm256_loadu_ps(imag + even_idx);
                const __m256 or_ = _mm256_loadu_ps(real + odd_idx);
                const __m256 oi = _mm256_loadu_ps(imag + odd_idx);

                // t = W * X[odd], with W conjugated for the inverse transform.
                __m256 tr = _mm256_mul_ps(wr, or_);
                __m256 ti = _mm256_mul_ps(wr, oi);
                if constexpr (Inverse) {
                    tr = _mm256_fmadd_ps(wi, oi, tr);    // wr*or + wi*oi
                    ti = _mm256_fnmadd_ps(wi, or_, ti);  // wr*oi - wi*or
                } else {
                    tr = _mm256_fnmadd_ps(wi, oi, tr);   // wr*or - wi*oi
                    ti = _mm256_fmadd_ps(wi, or_, ti);   // wr*oi + wi*or
                }

                _mm256_storeu_ps(real + even_idx, _mm256_add_ps(er, tr));
                _mm256_storeu_ps(imag + even_idx, _mm256_add_ps(ei, ti));
                _mm256_storeu_ps(real + odd_idx, _mm256_sub_ps(er, tr));
                _mm256_storeu_ps(imag + odd_idx, _mm256_sub_ps(ei, ti));
            }

            // Scalar tail (the first three stages, half_m < 8)
            for (; j < half_m; ++j) {
                const size_t even = k + j;
                const size_t odd = even + half_m;

                const float wr = tw_real[j];
                const float wi = Inverse ? -tw_imag[j] : tw_imag[j];

                const float tr = wr * real[odd] - wi * imag[odd];
                const float ti = wr * imag[odd] + wi * real[odd];

                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] = real[even] + tr;
                imag[even] = imag[even] + ti;
            }
        }
    }
}

} // anonymous namespace

void fft_rows_avx(float* real, float* imag, size_t rows, size_t cols,
                  bool inverse, float scale) {
    if (rows == 0 || cols < 2) {
        // cols == 1: the transform is the identity; only the scale remains.
        if (cols == 1 && scale != 1.0f) {
            scale_avx(real, imag, rows, scale);
        }
        return;
    }

    // One plan lookup for the whole batch (not one per row).
    const FFTPlanRef plan = fft_plan_for(cols);
    const bool use_avx = cols >= 16;

    for (size_t r = 0; r < rows; ++r) {
        float* row_real = real + r * cols;
        float* row_imag = imag + r * cols;

        if (!use_avx) {
            fft_scalar_core(row_real, row_imag, *plan, inverse);
        } else if (inverse) {
            fft_avx_core<true>(row_real, row_imag, *plan);
        } else {
            fft_avx_core<false>(row_real, row_imag, *plan);
        }
        if (scale != 1.0f) {
            scale_avx(row_real, row_imag, cols, scale);
        }
    }
}

} // namespace basalt::internal

namespace basalt::kernel {

namespace {

void fft_avx_impl(float* real, float* imag, size_t n, bool inverse, const char* who) {
    internal::validate_fft_args(real, imag, n, who);
    if (n < 2) {
        return;
    }
    // Small sizes take the scalar path inside fft_rows_avx.
    const float scale = inverse ? 1.0f / static_cast<float>(n) : 1.0f;
    internal::fft_rows_avx(real, imag, 1, n, inverse, scale);
}

} // anonymous namespace

void fft_forward_avx(float* real, float* imag, size_t n) {
    fft_avx_impl(real, imag, n, false, "fft_forward_avx");
}

void fft_inverse_avx(float* real, float* imag, size_t n) {
    fft_avx_impl(real, imag, n, true, "fft_inverse_avx");
}

} // namespace basalt::kernel
