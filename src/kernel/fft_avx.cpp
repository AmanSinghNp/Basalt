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

/// In-register 8x8 transpose of eight __m256 rows.
inline void transpose8x8_regs(__m256 (&r)[8]) {
    const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]);
    const __m256 t1 = _mm256_unpackhi_ps(r[0], r[1]);
    const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]);
    const __m256 t3 = _mm256_unpackhi_ps(r[2], r[3]);
    const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]);
    const __m256 t5 = _mm256_unpackhi_ps(r[4], r[5]);
    const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]);
    const __m256 t7 = _mm256_unpackhi_ps(r[6], r[7]);

    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    r[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
    r[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    r[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
    r[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    r[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
    r[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    r[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
    r[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

/// Radix-2 butterfly on register pairs: (a, b) -> (a + t, a - t).
inline void butterfly(__m256& ar, __m256& ai, __m256& br, __m256& bi,
                      __m256 tr, __m256 ti) {
    const __m256 nar = _mm256_add_ps(ar, tr);
    const __m256 nai = _mm256_add_ps(ai, ti);
    br = _mm256_sub_ps(ar, tr);
    bi = _mm256_sub_ps(ai, ti);
    ar = nar;
    ai = nai;
}

/// Stages 1-3 (block sizes 2, 4, 8) for n a multiple of 64, fully vectorised.
/// Each group of 64 consecutive complex values is viewed as an 8x8 matrix (one
/// 8-point block per row); an in-register transpose puts element e of eight
/// different blocks into register e, so all three stages become plain vector
/// butterflies with constant twiddles (1, W4, W8^k).
template <bool Inverse>
void first_three_stages_avx(float* real, float* imag, size_t n) {
    // |W8| components: cos(pi/4) = sin(pi/4).
    const __m256 c = _mm256_set1_ps(0.70710678118654752440f);

    for (size_t base = 0; base < n; base += 64) {
        __m256 re[8];
        __m256 im[8];
        for (size_t b = 0; b < 8; ++b) {
            re[b] = _mm256_loadu_ps(real + base + 8 * b);
            im[b] = _mm256_loadu_ps(imag + base + 8 * b);
        }
        transpose8x8_regs(re);
        transpose8x8_regs(im);

        // Stage 1: pairs (0,1) (2,3) (4,5) (6,7), twiddle 1.
        for (size_t e = 0; e < 8; e += 2) {
            butterfly(re[e], im[e], re[e + 1], im[e + 1], re[e + 1], im[e + 1]);
        }

        // Stage 2: pairs (0,2) (1,3) (4,6) (5,7); twiddles 1 and W4.
        // W4 = -i (forward) or +i (inverse): t = (xi, -xr) or (-xi, xr).
        for (size_t e = 0; e < 8; e += 4) {
            butterfly(re[e], im[e], re[e + 2], im[e + 2], re[e + 2], im[e + 2]);
            const __m256 xr = re[e + 3];
            const __m256 xi = im[e + 3];
            if constexpr (Inverse) {
                butterfly(re[e + 1], im[e + 1], re[e + 3], im[e + 3],
                          _mm256_sub_ps(_mm256_setzero_ps(), xi), xr);
            } else {
                butterfly(re[e + 1], im[e + 1], re[e + 3], im[e + 3],
                          xi, _mm256_sub_ps(_mm256_setzero_ps(), xr));
            }
        }

        // Stage 3: pairs (0,4) (1,5) (2,6) (3,7); twiddles 1, W8, W4, W8^3.
        butterfly(re[0], im[0], re[4], im[4], re[4], im[4]);
        {
            const __m256 xr = re[5];
            const __m256 xi = im[5];
            const __m256 sum = _mm256_add_ps(xr, xi);
            const __m256 dif = _mm256_sub_ps(xr, xi);
            if constexpr (Inverse) {   // W8 = c + c*i
                butterfly(re[1], im[1], re[5], im[5],
                          _mm256_mul_ps(c, dif), _mm256_mul_ps(c, sum));
            } else {                   // W8 = c - c*i
                butterfly(re[1], im[1], re[5], im[5],
                          _mm256_mul_ps(c, sum), _mm256_mul_ps(c, _mm256_sub_ps(_mm256_setzero_ps(), dif)));
            }
        }
        {
            const __m256 xr = re[6];
            const __m256 xi = im[6];
            if constexpr (Inverse) {
                butterfly(re[2], im[2], re[6], im[6],
                          _mm256_sub_ps(_mm256_setzero_ps(), xi), xr);
            } else {
                butterfly(re[2], im[2], re[6], im[6],
                          xi, _mm256_sub_ps(_mm256_setzero_ps(), xr));
            }
        }
        {
            const __m256 xr = re[7];
            const __m256 xi = im[7];
            const __m256 sum = _mm256_add_ps(xr, xi);
            const __m256 dif = _mm256_sub_ps(xr, xi);
            if constexpr (Inverse) {   // W8^3 = -c + c*i
                butterfly(re[3], im[3], re[7], im[7],
                          _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(c, sum)),
                          _mm256_mul_ps(c, dif));
            } else {                   // W8^3 = -c - c*i
                butterfly(re[3], im[3], re[7], im[7],
                          _mm256_mul_ps(c, _mm256_sub_ps(_mm256_setzero_ps(), dif)),
                          _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(c, sum)));
            }
        }

        transpose8x8_regs(re);
        transpose8x8_regs(im);
        for (size_t b = 0; b < 8; ++b) {
            _mm256_storeu_ps(real + base + 8 * b, re[b]);
            _mm256_storeu_ps(imag + base + 8 * b, im[b]);
        }
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
    // Stages 1-3 have half_m < 8 and no room for a vector across j; for
    // n >= 64 they run as a fused, fully vectorised pass (see above).
    size_t first_stage = 1;
    if (n >= 64) {
        first_three_stages_avx<Inverse>(real, imag, n);
        first_stage = 4;
    }
    for (size_t s = first_stage; s <= plan.log2n; ++s) {
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
