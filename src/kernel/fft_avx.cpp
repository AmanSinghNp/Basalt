#include "basalt/kernel/fft_avx.hpp"
#include "basalt/internal/fft_plan.hpp"
#include "basalt/internal/simd_dispatch.hpp"
#include "basalt/kernel/cpu_features.hpp"

#include <immintrin.h>
#include <utility>

namespace basalt::internal {

namespace {

/// Multiply n elements of real/imag by 'scale' (AVX2).
BASALT_TARGET_AVX2 void scale_avx(float* real, float* imag, size_t n, float scale_val) {
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
BASALT_TARGET_AVX2 inline void transpose8x8_regs(__m256 (&r)[8]) {
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
BASALT_TARGET_AVX2 inline void butterfly(__m256& ar, __m256& ai, __m256& br, __m256& bi, __m256 tr,
                                         __m256 ti) {
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
BASALT_TARGET_AVX2 void first_three_stages_avx(float* real, float* imag, size_t n) {
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
                butterfly(re[e + 1], im[e + 1], re[e + 3], im[e + 3], xi,
                          _mm256_sub_ps(_mm256_setzero_ps(), xr));
            }
        }

        // Stage 3: pairs (0,4) (1,5) (2,6) (3,7); twiddles 1, W8, W4, W8^3.
        butterfly(re[0], im[0], re[4], im[4], re[4], im[4]);
        {
            const __m256 xr = re[5];
            const __m256 xi = im[5];
            const __m256 sum = _mm256_add_ps(xr, xi);
            const __m256 dif = _mm256_sub_ps(xr, xi);
            if constexpr (Inverse) { // W8 = c + c*i
                butterfly(re[1], im[1], re[5], im[5], _mm256_mul_ps(c, dif), _mm256_mul_ps(c, sum));
            } else { // W8 = c - c*i
                butterfly(re[1], im[1], re[5], im[5], _mm256_mul_ps(c, sum),
                          _mm256_mul_ps(c, _mm256_sub_ps(_mm256_setzero_ps(), dif)));
            }
        }
        {
            const __m256 xr = re[6];
            const __m256 xi = im[6];
            if constexpr (Inverse) {
                butterfly(re[2], im[2], re[6], im[6], _mm256_sub_ps(_mm256_setzero_ps(), xi), xr);
            } else {
                butterfly(re[2], im[2], re[6], im[6], xi, _mm256_sub_ps(_mm256_setzero_ps(), xr));
            }
        }
        {
            const __m256 xr = re[7];
            const __m256 xi = im[7];
            const __m256 sum = _mm256_add_ps(xr, xi);
            const __m256 dif = _mm256_sub_ps(xr, xi);
            if constexpr (Inverse) { // W8^3 = -c + c*i
                butterfly(re[3], im[3], re[7], im[7],
                          _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(c, sum)),
                          _mm256_mul_ps(c, dif));
            } else { // W8^3 = -c - c*i
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

/// Unscaled AVX2/FMA transform for n >= 16 (BASALT_TARGET_AVX2: run only when
/// the CPU supports AVX2 and FMA). 'Inverse' conjugates the twiddles.
template <bool Inverse>
BASALT_TARGET_AVX2 void fft_avx_core(float* real, float* imag, const FFTPlan& plan) {
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
                    tr = _mm256_fmadd_ps(wi, oi, tr);   // wr*or + wi*oi
                    ti = _mm256_fnmadd_ps(wi, or_, ti); // wr*oi - wi*or
                } else {
                    tr = _mm256_fnmadd_ps(wi, oi, tr); // wr*or - wi*oi
                    ti = _mm256_fmadd_ps(wi, or_, ti); // wr*oi + wi*or
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

/// Multiply n elements of real/imag by 'scale' (portable).
void scale_scalar(float* real, float* imag, size_t n, float scale_val) {
    for (size_t i = 0; i < n; ++i) {
        real[i] *= scale_val;
        imag[i] *= scale_val;
    }
}

/// Row loop of the AVX2 path. The whole batch runs inside one target("avx2,fma")
/// function so the kernels above inline into it.
BASALT_TARGET_AVX2 void fft_rows_avx2_impl(float* real, float* imag, size_t rows, size_t cols,
                                           bool inverse, float scale, const FFTPlan& plan) {
    const bool use_avx = cols >= 16;

    for (size_t r = 0; r < rows; ++r) {
        float* row_real = real + r * cols;
        float* row_imag = imag + r * cols;

        if (!use_avx) {
            fft_scalar_core(row_real, row_imag, plan, inverse);
        } else if (inverse) {
            fft_avx_core<true>(row_real, row_imag, plan);
        } else {
            fft_avx_core<false>(row_real, row_imag, plan);
        }
        if (scale != 1.0f) {
            scale_avx(row_real, row_imag, cols, scale);
        }
    }
}

/// Row loop of the portable path (CPUs without AVX2/FMA, or SimdMode capped).
void fft_rows_scalar_impl(float* real, float* imag, size_t rows, size_t cols, bool inverse,
                          float scale, const FFTPlan& plan) {
    for (size_t r = 0; r < rows; ++r) {
        float* row_real = real + r * cols;
        float* row_imag = imag + r * cols;
        fft_scalar_core(row_real, row_imag, plan, inverse);
        if (scale != 1.0f) {
            scale_scalar(row_real, row_imag, cols, scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Column FFTs without transposes.
//
// A strip of 'width' adjacent columns is gathered into a contiguous buffer
// (rows in bit-reversed order), transformed with a radix-2 DIT FFT whose SIMD
// lanes are the columns, and scattered back (scaled) to its place. Every
// butterfly then works on whole rows of the buffer with contiguous 8-float
// loads, and the matrix itself is read and written exactly once.
//
// The buffer is split into sub-strips of 8*V columns, each stored as its own
// contiguous [rows][8*V] block, so the early stages can run cache-blocked on
// a few rows at a time while the gather/scatter still touch whole cache lines
// (and few pages) per matrix row.
// ---------------------------------------------------------------------------

/// t = w * x, with w conjugated for the inverse transform.
template <bool Inverse>
BASALT_TARGET_AVX2 inline void cmul(__m256 wr, __m256 wi, __m256 xr, __m256 xi, __m256& tr,
                                    __m256& ti) {
    tr = _mm256_mul_ps(wr, xr);
    ti = _mm256_mul_ps(wr, xi);
    if constexpr (Inverse) {
        tr = _mm256_fmadd_ps(wi, xi, tr);  // wr*xr + wi*xi
        ti = _mm256_fnmadd_ps(wi, xr, ti); // wr*xi - wi*xr
    } else {
        tr = _mm256_fnmadd_ps(wi, xi, tr); // wr*xr - wi*xi
        ti = _mm256_fmadd_ps(wi, xr, ti);  // wr*xi + wi*xr
    }
}

/// Stages 1 and 2 fused (all twiddles are 1 or +-i) on every 4-row group of
/// a [nrows][8*V] block.
template <bool Inverse, size_t V>
BASALT_TARGET_AVX2 void col_stages_12(float* br, float* bi, size_t nrows) {
    constexpr size_t W = 8 * V;
    for (size_t k = 0; k < nrows; k += 4) {
        float* pr = br + k * W;
        float* pi = bi + k * W;
        for (size_t v = 0; v < V; ++v) {
            const __m256 a0r = _mm256_loadu_ps(pr + 8 * v);
            const __m256 a0i = _mm256_loadu_ps(pi + 8 * v);
            const __m256 a1r = _mm256_loadu_ps(pr + W + 8 * v);
            const __m256 a1i = _mm256_loadu_ps(pi + W + 8 * v);
            const __m256 a2r = _mm256_loadu_ps(pr + 2 * W + 8 * v);
            const __m256 a2i = _mm256_loadu_ps(pi + 2 * W + 8 * v);
            const __m256 a3r = _mm256_loadu_ps(pr + 3 * W + 8 * v);
            const __m256 a3i = _mm256_loadu_ps(pi + 3 * W + 8 * v);

            const __m256 b0r = _mm256_add_ps(a0r, a1r), b0i = _mm256_add_ps(a0i, a1i);
            const __m256 b1r = _mm256_sub_ps(a0r, a1r), b1i = _mm256_sub_ps(a0i, a1i);
            const __m256 b2r = _mm256_add_ps(a2r, a3r), b2i = _mm256_add_ps(a2i, a3i);
            const __m256 b3r = _mm256_sub_ps(a2r, a3r), b3i = _mm256_sub_ps(a2i, a3i);

            // Stage 2 twiddles: 1 for (b0, b2); W4 = -i (forward) / +i (inverse) for (b1, b3).
            __m256 rr, ri; // W4 * b3
            if constexpr (Inverse) {
                rr = _mm256_sub_ps(_mm256_setzero_ps(), b3i);
                ri = b3r;
            } else {
                rr = b3i;
                ri = _mm256_sub_ps(_mm256_setzero_ps(), b3r);
            }
            _mm256_storeu_ps(pr + 8 * v, _mm256_add_ps(b0r, b2r));
            _mm256_storeu_ps(pi + 8 * v, _mm256_add_ps(b0i, b2i));
            _mm256_storeu_ps(pr + W + 8 * v, _mm256_add_ps(b1r, rr));
            _mm256_storeu_ps(pi + W + 8 * v, _mm256_add_ps(b1i, ri));
            _mm256_storeu_ps(pr + 2 * W + 8 * v, _mm256_sub_ps(b0r, b2r));
            _mm256_storeu_ps(pi + 2 * W + 8 * v, _mm256_sub_ps(b0i, b2i));
            _mm256_storeu_ps(pr + 3 * W + 8 * v, _mm256_sub_ps(b1r, rr));
            _mm256_storeu_ps(pi + 3 * W + 8 * v, _mm256_sub_ps(b1i, ri));
        }
    }
}

/// One radix-2 stage s (block size 2^s) on a [nrows][8*V] block; nrows is a
/// multiple of 2^s.
template <bool Inverse, size_t V>
BASALT_TARGET_AVX2 void col_stage_r2(float* br, float* bi, size_t nrows, size_t s,
                                     const FFTPlan& plan) {
    constexpr size_t W = 8 * V;
    const size_t half_m = size_t{1} << (s - 1);
    const float* tw_real = plan.tw_real.data() + (half_m - 1);
    const float* tw_imag = plan.tw_imag.data() + (half_m - 1);

    for (size_t k = 0; k < nrows; k += 2 * half_m) {
        for (size_t j = 0; j < half_m; ++j) {
            const __m256 wr = _mm256_set1_ps(tw_real[j]);
            const __m256 wi = _mm256_set1_ps(tw_imag[j]);
            float* er = br + (k + j) * W;
            float* ei = bi + (k + j) * W;
            float* odr = er + half_m * W;
            float* odi = ei + half_m * W;
            for (size_t v = 0; v < V; ++v) {
                const __m256 xr = _mm256_loadu_ps(er + 8 * v);
                const __m256 xi = _mm256_loadu_ps(ei + 8 * v);
                __m256 tr, ti;
                cmul<Inverse>(wr, wi, _mm256_loadu_ps(odr + 8 * v), _mm256_loadu_ps(odi + 8 * v),
                              tr, ti);
                _mm256_storeu_ps(er + 8 * v, _mm256_add_ps(xr, tr));
                _mm256_storeu_ps(ei + 8 * v, _mm256_add_ps(xi, ti));
                _mm256_storeu_ps(odr + 8 * v, _mm256_sub_ps(xr, tr));
                _mm256_storeu_ps(odi + 8 * v, _mm256_sub_ps(xi, ti));
            }
        }
    }
}

/// Stages s and s+1 fused into one pass (radix-2^2) on a [nrows][8*V] block;
/// nrows is a multiple of 2^(s+1). Same arithmetic as two radix-2 stages, but
/// every element is loaded and stored once instead of twice.
template <bool Inverse, size_t V>
BASALT_TARGET_AVX2 void col_stages_r4(float* br, float* bi, size_t nrows, size_t s,
                                      const FFTPlan& plan) {
    constexpr size_t W = 8 * V;
    const size_t h = size_t{1} << (s - 1); // half of stage s
    const float* w_real = plan.tw_real.data() + (h - 1);
    const float* w_imag = plan.tw_imag.data() + (h - 1);
    const float* u_real = plan.tw_real.data() + (2 * h - 1); // stage s+1, entries 0..h-1
    const float* u_imag = plan.tw_imag.data() + (2 * h - 1);

    for (size_t k = 0; k < nrows; k += 4 * h) {
        for (size_t j = 0; j < h; ++j) {
            const __m256 wr = _mm256_set1_ps(w_real[j]);
            const __m256 wi = _mm256_set1_ps(w_imag[j]);
            const __m256 ur = _mm256_set1_ps(u_real[j]);
            const __m256 ui = _mm256_set1_ps(u_imag[j]);
            float* p0r = br + (k + j) * W;
            float* p0i = bi + (k + j) * W;
            const size_t hs = h * W;
            for (size_t v = 0; v < V; ++v) {
                const __m256 a0r = _mm256_loadu_ps(p0r + 8 * v);
                const __m256 a0i = _mm256_loadu_ps(p0i + 8 * v);
                __m256 t1r, t1i, t3r, t3i;
                cmul<Inverse>(wr, wi, _mm256_loadu_ps(p0r + hs + 8 * v),
                              _mm256_loadu_ps(p0i + hs + 8 * v), t1r, t1i);
                const __m256 a2r = _mm256_loadu_ps(p0r + 2 * hs + 8 * v);
                const __m256 a2i = _mm256_loadu_ps(p0i + 2 * hs + 8 * v);
                cmul<Inverse>(wr, wi, _mm256_loadu_ps(p0r + 3 * hs + 8 * v),
                              _mm256_loadu_ps(p0i + 3 * hs + 8 * v), t3r, t3i);

                // Stage s: (a0, a1) -> (b0, b1), (a2, a3) -> (b2, b3).
                const __m256 b0r = _mm256_add_ps(a0r, t1r), b0i = _mm256_add_ps(a0i, t1i);
                const __m256 b1r = _mm256_sub_ps(a0r, t1r), b1i = _mm256_sub_ps(a0i, t1i);
                const __m256 b2r = _mm256_add_ps(a2r, t3r), b2i = _mm256_add_ps(a2i, t3i);
                const __m256 b3r = _mm256_sub_ps(a2r, t3r), b3i = _mm256_sub_ps(a2i, t3i);

                // Stage s+1: (b0, b2) with u, (b1, b3) with u * W4.
                __m256 rr, ri; // W4 * b3
                if constexpr (Inverse) {
                    rr = _mm256_sub_ps(_mm256_setzero_ps(), b3i);
                    ri = b3r;
                } else {
                    rr = b3i;
                    ri = _mm256_sub_ps(_mm256_setzero_ps(), b3r);
                }
                __m256 q2r, q2i, q3r, q3i;
                cmul<Inverse>(ur, ui, b2r, b2i, q2r, q2i);
                cmul<Inverse>(ur, ui, rr, ri, q3r, q3i);

                _mm256_storeu_ps(p0r + 8 * v, _mm256_add_ps(b0r, q2r));
                _mm256_storeu_ps(p0i + 8 * v, _mm256_add_ps(b0i, q2i));
                _mm256_storeu_ps(p0r + hs + 8 * v, _mm256_add_ps(b1r, q3r));
                _mm256_storeu_ps(p0i + hs + 8 * v, _mm256_add_ps(b1i, q3i));
                _mm256_storeu_ps(p0r + 2 * hs + 8 * v, _mm256_sub_ps(b0r, q2r));
                _mm256_storeu_ps(p0i + 2 * hs + 8 * v, _mm256_sub_ps(b0i, q2i));
                _mm256_storeu_ps(p0r + 3 * hs + 8 * v, _mm256_sub_ps(b1r, q3r));
                _mm256_storeu_ps(p0i + 3 * hs + 8 * v, _mm256_sub_ps(b1i, q3i));
            }
        }
    }
}

/// Stages s_from..s_to (inclusive) on a [nrows][8*V] block, two stages per
/// pass where possible. nrows is a multiple of 2^s_to.
template <bool Inverse, size_t V>
BASALT_TARGET_AVX2 void col_stages(float* br, float* bi, size_t nrows, size_t s_from, size_t s_to,
                                   const FFTPlan& plan) {
    size_t s = s_from;
    if (s == 1) {
        if (s_to >= 2) {
            col_stages_12<Inverse, V>(br, bi, nrows);
            s = 3;
        } else {
            col_stage_r2<Inverse, V>(br, bi, nrows, 1, plan);
            s = 2;
        }
    }
    for (; s + 1 <= s_to; s += 2) {
        col_stages_r4<Inverse, V>(br, bi, nrows, s, plan);
    }
    if (s <= s_to) {
        col_stage_r2<Inverse, V>(br, bi, nrows, s, plan);
    }
}

/// Transform all columns of one [plan.n][8*V] sub-strip whose rows are already
/// in bit-reversed order. Stages 1..block_log2 run block by block (a block is
/// 2^block_log2 rows, sized to stay in L1); the rest run over the whole strip.
template <bool Inverse, size_t V>
BASALT_TARGET_AVX2 void col_fft_substrip_avx(float* br, float* bi, const FFTPlan& plan,
                                             size_t block_log2) {
    constexpr size_t W = 8 * V;
    const size_t log2n = plan.log2n;
    const size_t lb = block_log2 < log2n ? block_log2 : log2n;
    const size_t block_rows = size_t{1} << lb;
    for (size_t k = 0; k < plan.n; k += block_rows) {
        col_stages<Inverse, V>(br + k * W, bi + k * W, block_rows, 1, lb, plan);
    }
    if (lb < log2n) {
        col_stages<Inverse, V>(br, bi, plan.n, lb + 1, log2n, plan);
    }
}

/// Strip driver of the AVX2 path: all cols/width strips of the matrix. The
/// buffer holds width/(8*V) sub-strips of rows * 8*V floats each.
template <bool Inverse, size_t V>
BASALT_TARGET_AVX2 void fft_cols_strips_avx2_impl(float* real, float* imag, size_t rows,
                                                  size_t cols, size_t width, float scale,
                                                  float* buf_r, float* buf_i, const FFTPlan& plan) {
    constexpr size_t SW = 8 * V;
    const size_t nsub = width / SW;
    const __m256 vscale = _mm256_set1_ps(scale);
    const bool scaled = scale != 1.0f;

    // Largest block (in rows) whose sub-strip data (re + im) stays within ~16 KiB.
    size_t block_log2 = 1;
    while ((size_t{1} << (block_log2 + 1)) * SW * 2 * sizeof(float) <= 16384) {
        ++block_log2;
    }

    for (size_t c0 = 0; c0 < cols; c0 += width) {
        // Gather: sub-strip q, row i = matrix row bit_reverse(i).
        size_t rev = 0;
        for (size_t i = 0; i < rows; ++i) {
            const float* sr = real + rev * cols + c0;
            const float* si = imag + rev * cols + c0;
            for (size_t q = 0; q < nsub; ++q) {
                float* dr = buf_r + (q * rows + i) * SW;
                float* di = buf_i + (q * rows + i) * SW;
                for (size_t v = 0; v < V; ++v) {
                    _mm256_storeu_ps(dr + 8 * v, _mm256_loadu_ps(sr + q * SW + 8 * v));
                    _mm256_storeu_ps(di + 8 * v, _mm256_loadu_ps(si + q * SW + 8 * v));
                }
            }
            size_t bit = rows >> 1;
            while (rev & bit) {
                rev ^= bit;
                bit >>= 1;
            }
            rev |= bit;
        }

        for (size_t q = 0; q < nsub; ++q) {
            col_fft_substrip_avx<Inverse, V>(buf_r + q * rows * SW, buf_i + q * rows * SW, plan,
                                             block_log2);
        }

        // Scatter back (natural order), applying the scale.
        for (size_t i = 0; i < rows; ++i) {
            float* dr = real + i * cols + c0;
            float* di = imag + i * cols + c0;
            for (size_t q = 0; q < nsub; ++q) {
                const float* sr = buf_r + (q * rows + i) * SW;
                const float* si = buf_i + (q * rows + i) * SW;
                for (size_t v = 0; v < V; ++v) {
                    __m256 xr = _mm256_loadu_ps(sr + 8 * v);
                    __m256 xi = _mm256_loadu_ps(si + 8 * v);
                    if (scaled) {
                        xr = _mm256_mul_ps(xr, vscale);
                        xi = _mm256_mul_ps(xi, vscale);
                    }
                    _mm256_storeu_ps(dr + q * SW + 8 * v, xr);
                    _mm256_storeu_ps(di + q * SW + 8 * v, xi);
                }
            }
        }
    }
}

template <bool Inverse>
void fft_cols_strips_avx2_dispatch(float* real, float* imag, size_t rows, size_t cols, size_t width,
                                   float scale, float* buf_r, float* buf_i, const FFTPlan& plan) {
    // Sub-strip of 32 lanes when the strip allows it (measured fastest), else 16 or 8.
    if (width >= 32) {
        fft_cols_strips_avx2_impl<Inverse, 4>(real, imag, rows, cols, width, scale, buf_r, buf_i,
                                              plan);
    } else if (width >= 16) {
        fft_cols_strips_avx2_impl<Inverse, 2>(real, imag, rows, cols, width, scale, buf_r, buf_i,
                                              plan);
    } else {
        fft_cols_strips_avx2_impl<Inverse, 1>(real, imag, rows, cols, width, scale, buf_r, buf_i,
                                              plan);
    }
}

/// Portable strip driver (CPUs without AVX2/FMA, or SimdMode capped). Same
/// algorithm and operation order as the AVX2 path, lane by lane.
template <bool Inverse>
void fft_cols_strips_scalar_impl(float* real, float* imag, size_t rows, size_t cols, size_t W,
                                 float scale, float* buf_r, float* buf_i, const FFTPlan& plan) {
    for (size_t c0 = 0; c0 < cols; c0 += W) {
        size_t rev = 0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t l = 0; l < W; ++l) {
                buf_r[i * W + l] = real[rev * cols + c0 + l];
                buf_i[i * W + l] = imag[rev * cols + c0 + l];
            }
            size_t bit = rows >> 1;
            while (rev & bit) {
                rev ^= bit;
                bit >>= 1;
            }
            rev |= bit;
        }

        for (size_t s = 1; s <= plan.log2n; ++s) {
            const size_t m = size_t{1} << s;
            const size_t half_m = m >> 1;
            const float* tw_real = plan.tw_real.data() + (half_m - 1);
            const float* tw_imag = plan.tw_imag.data() + (half_m - 1);
            for (size_t k = 0; k < rows; k += m) {
                for (size_t j = 0; j < half_m; ++j) {
                    const float wr = tw_real[j];
                    const float wi = Inverse ? -tw_imag[j] : tw_imag[j];
                    float* er = buf_r + (k + j) * W;
                    float* ei = buf_i + (k + j) * W;
                    float* odr = er + half_m * W;
                    float* odi = ei + half_m * W;
                    for (size_t l = 0; l < W; ++l) {
                        const float tr = wr * odr[l] - wi * odi[l];
                        const float ti = wr * odi[l] + wi * odr[l];
                        odr[l] = er[l] - tr;
                        odi[l] = ei[l] - ti;
                        er[l] = er[l] + tr;
                        ei[l] = ei[l] + ti;
                    }
                }
            }
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t l = 0; l < W; ++l) {
                real[i * cols + c0 + l] = buf_r[i * W + l] * scale;
                imag[i * cols + c0 + l] = buf_i[i * W + l] * scale;
            }
        }
    }
}

} // anonymous namespace

void fft_rows_avx(float* real, float* imag, size_t rows, size_t cols, bool inverse, float scale) {
    // One dispatch decision per call (not per row).
    const bool avx2 = kernel::detail::use_avx2();

    if (rows == 0 || cols < 2) {
        // cols == 1: the transform is the identity; only the scale remains.
        if (cols == 1 && scale != 1.0f) {
            if (avx2) {
                scale_avx(real, imag, rows, scale);
            } else {
                scale_scalar(real, imag, rows, scale);
            }
        }
        return;
    }

    // One plan lookup for the whole batch (not one per row).
    const FFTPlanRef plan = fft_plan_for(cols);
    if (avx2) {
        fft_rows_avx2_impl(real, imag, rows, cols, inverse, scale, *plan);
    } else {
        fft_rows_scalar_impl(real, imag, rows, cols, inverse, scale, *plan);
    }
}

size_t fft_cols_strip_width(size_t rows, size_t cols) {
    // 64 columns (256 bytes of every touched matrix row) amortises the strided
    // row accesses best on 256..4096-wide matrices; narrower strips lose to
    // page-walk and cache-line overhead, wider ones to cache capacity.
    (void)rows;
    constexpr size_t kStripWidth = 64;
    return cols < kStripWidth ? cols : kStripWidth;
}

void fft_cols_strips(float* real, float* imag, size_t rows, size_t cols, bool inverse, float scale,
                     float* buf_real, float* buf_imag, size_t width) {
    if (rows < 2) {
        // Length-1 column transform is the identity; only the scale remains.
        if (scale != 1.0f) {
            for (size_t i = 0; i < cols; ++i) {
                real[i] *= scale;
                imag[i] *= scale;
            }
        }
        return;
    }
    const FFTPlanRef plan = fft_plan_for(rows);
    // One dispatch decision per call. The vector path needs whole 8-lane groups.
    if (kernel::detail::use_avx2() && width >= 8 && width % 8 == 0) {
        if (inverse) {
            fft_cols_strips_avx2_dispatch<true>(real, imag, rows, cols, width, scale, buf_real,
                                                buf_imag, *plan);
        } else {
            fft_cols_strips_avx2_dispatch<false>(real, imag, rows, cols, width, scale, buf_real,
                                                 buf_imag, *plan);
        }
    } else if (inverse) {
        fft_cols_strips_scalar_impl<true>(real, imag, rows, cols, width, scale, buf_real, buf_imag,
                                          *plan);
    } else {
        fft_cols_strips_scalar_impl<false>(real, imag, rows, cols, width, scale, buf_real, buf_imag,
                                           *plan);
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
