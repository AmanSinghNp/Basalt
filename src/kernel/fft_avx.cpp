#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/fft.hpp"
#include "basalt/kernel/memory_hints.hpp"

#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace basalt::kernel {

namespace {

struct TwiddleStage {
    std::vector<float> real;
    std::vector<float> imag;
};

struct TwiddlePlan {
    std::vector<TwiddleStage> stages;
};

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

const TwiddlePlan& twiddle_plan_for(size_t n, bool inverse) {
    thread_local std::unordered_map<size_t, TwiddlePlan> forward_cache;
    thread_local std::unordered_map<size_t, TwiddlePlan> inverse_cache;

    auto& cache = inverse ? inverse_cache : forward_cache;
    auto it = cache.find(n);
    if (it != cache.end()) {
        return it->second;
    }

    TwiddlePlan plan;
    const size_t log2n = log2_of(n);
    plan.stages.reserve(log2n + 1);
    plan.stages.emplace_back();

    for (size_t s = 1; s <= log2n; ++s) {
        const size_t m = 1u << s;
        const size_t half_m = m >> 1;
        const double angle_step = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(m);

        TwiddleStage stage;
        stage.real.resize(half_m);
        stage.imag.resize(half_m);
        for (size_t j = 0; j < half_m; ++j) {
            const double angle = angle_step * static_cast<double>(j);
            stage.real[j] = static_cast<float>(std::cos(angle));
            stage.imag[j] = static_cast<float>(std::sin(angle));
        }
        plan.stages.push_back(std::move(stage));
    }

    return cache.emplace(n, std::move(plan)).first->second;
}

/// AVX2/FMA core FFT implementation.
void fft_avx_impl(float* real, float* imag, size_t n, bool inverse) {
    assert(n > 0 && (n & (n - 1)) == 0 && "n must be a power of 2");

    // Fallback to scalar for small sizes where AVX overhead isn't worth it
    if (n < 16) {
        if (inverse) {
            fft_inverse(real, imag, n);
        } else {
            fft_forward(real, imag, n);
        }
        return;
    }

    size_t log2n = log2_of(n);
    const TwiddlePlan& twiddles = twiddle_plan_for(n, inverse);

    // --- Step 1: Bit-reversal permutation (memory-bound, scalar is fine) ---
    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, log2n);
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // --- Step 2: Butterfly stages ---
    for (size_t s = 1; s <= log2n; ++s) {
        size_t m = 1u << s;
        size_t half_m = m >> 1;
        const TwiddleStage& stage = twiddles.stages[s];
        const float* tw_real = stage.real.data();
        const float* tw_imag = stage.imag.data();

        // Process each block
        for (size_t k = 0; k < n; k += m) {
            size_t j = 0;

            // AVX2 path: process 8 butterflies at once
            // Only possible when half_m >= 8 (stage >= 4, m >= 16)
            if (half_m >= 8) {
                for (; j + 7 < half_m; j += 8) {
                    size_t even_idx = k + j;
                    size_t odd_idx  = k + j + half_m;

                    // Prefetch next iteration's data
                    if (j + 15 < half_m) {
                        prefetch_l1(real + k + j + 8);
                        prefetch_l1(imag + k + j + 8);
                        prefetch_l1(real + k + j + 8 + half_m);
                        prefetch_l1(imag + k + j + 8 + half_m);
                    }

                    // Load twiddle factors
                    __m256 wr = _mm256_loadu_ps(tw_real + j);
                    __m256 wi = _mm256_loadu_ps(tw_imag + j);

                    // Load even and odd elements
                    __m256 er = _mm256_loadu_ps(real + even_idx);
                    __m256 ei = _mm256_loadu_ps(imag + even_idx);
                    __m256 or_ = _mm256_loadu_ps(real + odd_idx);
                    __m256 oi = _mm256_loadu_ps(imag + odd_idx);

                    // Complex multiply: t = W * X[odd]
                    // tr = wr*or - wi*oi  → use FMA: tr = fmsub(wr, or, wi*oi)
                    //                       but we can use: tr = fmadd(wr, or, -wi*oi)
                    //                       = wr*or + (-wi)*oi  → fnmadd for second term
                    // First: wr * or_
                    __m256 tr = _mm256_mul_ps(wr, or_);
                    // tr = tr - wi * oi   →  FMA: tr = fnmadd(wi, oi, tr) = tr - wi*oi
                    tr = _mm256_fnmadd_ps(wi, oi, tr);

                    // ti = wr*oi + wi*or_  →  FMA: ti = fmadd(wi, or_, wr*oi)
                    __m256 ti = _mm256_mul_ps(wr, oi);
                    ti = _mm256_fmadd_ps(wi, or_, ti);

                    // Butterfly: even = even + t,  odd = even - t
                    __m256 new_er = _mm256_add_ps(er, tr);
                    __m256 new_ei = _mm256_add_ps(ei, ti);
                    __m256 new_or = _mm256_sub_ps(er, tr);
                    __m256 new_oi = _mm256_sub_ps(ei, ti);

                    // Store results
                    _mm256_storeu_ps(real + even_idx, new_er);
                    _mm256_storeu_ps(imag + even_idx, new_ei);
                    _mm256_storeu_ps(real + odd_idx, new_or);
                    _mm256_storeu_ps(imag + odd_idx, new_oi);
                }
            }

            // Scalar tail for remaining butterflies
            for (; j < half_m; ++j) {
                size_t even = k + j;
                size_t odd  = k + j + half_m;

                float wr = tw_real[j];
                float wi = tw_imag[j];

                float tr = wr * real[odd] - wi * imag[odd];
                float ti = wr * imag[odd] + wi * real[odd];

                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] = real[even] + tr;
                imag[even] = imag[even] + ti;
            }
        }
    }

    // --- Step 3: Scale by 1/N for inverse (AVX2 vectorized) ---
    if (inverse) {
        float scale_val = 1.0f / static_cast<float>(n);
        __m256 scale = _mm256_set1_ps(scale_val);

        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 r = _mm256_loadu_ps(real + i);
            __m256 im = _mm256_loadu_ps(imag + i);
            _mm256_storeu_ps(real + i, _mm256_mul_ps(r, scale));
            _mm256_storeu_ps(imag + i, _mm256_mul_ps(im, scale));
        }
        // Scalar tail
        for (; i < n; ++i) {
            real[i] *= scale_val;
            imag[i] *= scale_val;
        }
    }
}

} // anonymous namespace

void fft_forward_avx(float* real, float* imag, size_t n) {
    fft_avx_impl(real, imag, n, false);
}

void fft_inverse_avx(float* real, float* imag, size_t n) {
    fft_avx_impl(real, imag, n, true);
}

} // namespace basalt::kernel
