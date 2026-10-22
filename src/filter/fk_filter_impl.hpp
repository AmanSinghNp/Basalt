#pragma once

#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/cpu_features.hpp"
#include "basalt/kernel/simd_mode.hpp"
#include "basalt/utils/fast_math.hpp"

#include <cstddef>
#include <immintrin.h>

namespace basalt::filter::detail {

struct FKFilterExecutionPlan {
    size_t tile_rows = 64;
    size_t tile_cols = 128;
    size_t prefetch_distance = 16;
};

FKFilterExecutionPlan make_execution_plan(size_t rows, size_t cols);

/// Throw std::invalid_argument for unusable parameters / buffers.
void validate_params(const FKParams& params);
void validate_buffer(const basalt::kernel::ComplexSoA& data, size_t rows, size_t cols);

/// Portable scalar f-k filter (float arithmetic, SSE2 baseline). Same weight
/// formula as the AVX paths but evaluated with std::exp instead of the
/// polynomial approximation, so weights agree with the AVX paths to ~1e-5.
void apply_fk_filter_scalar(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                            const FKParams& params);

// The AVX2 / AVX-512 entry points below require AVX2+FMA (and AVX-512F/DQ for
// the latter). apply_fk_filter dispatches on cpu_features(); never call them
// directly without that check.
void apply_fk_filter_avx2(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                          const FKParams& params, const FKFilterExecutionPlan& plan);

void apply_fk_filter_avx512(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                            const FKParams& params, const FKFilterExecutionPlan& plan);

// Signed FFT-order index to frequency/wavenumber. Indices 0..size/2 are
// non-negative; the rest wrap to negative values. For even `size` the
// Nyquist bin (index == size/2) is reported as +Nyquist.
inline float wrapped_frequency(size_t index, size_t size, double step) {
    if (index <= size / 2) {
        return static_cast<float>(static_cast<double>(index) * step);
    }
    return static_cast<float>((static_cast<double>(index) - static_cast<double>(size)) * step);
}

inline float wrapped_wavenumber(size_t index, size_t size, double step) {
    return wrapped_frequency(index, size, step);
}

/// True for the Nyquist bin of an even-length axis. Its sign is ambiguous
/// (+Nyquist and -Nyquist alias), so the weight there is symmetrised to keep
/// H(f,k) == H(-f,-k) exact for asymmetric velocity bands.
inline bool is_nyquist_index(size_t index, size_t size) {
    return size % 2 == 0 && index == size / 2;
}

/// Per-call constants of the signed velocity band [vmin, vmax].
struct FKBandConsts {
    __m256 steep; // 10 / taper_width
    __m256 vmin;
    __m256 vmax;
};

BASALT_TARGET_AVX2 inline FKBandConsts make_band_consts(const FKParams& params) {
    FKBandConsts c;
    c.steep = _mm256_set1_ps(10.0f / static_cast<float>(params.taper_width));
    c.vmin = _mm256_set1_ps(static_cast<float>(params.mute_vel_min));
    c.vmax = _mm256_set1_ps(static_cast<float>(params.mute_vel_max));
    return c;
}

/// Pass weight for signed apparent velocity v: 1 - S((v-min)s) * S((max-v)s),
/// i.e. ~0 inside [min, max] and ~1 outside, with sigmoid tapers at both edges.
BASALT_TARGET_AVX2 inline __m256 band_weight_ps(__m256 vel, const FKBandConsts& c) {
    const __m256 lo =
        basalt::utils::fast_sigmoid_avx(_mm256_mul_ps(_mm256_sub_ps(vel, c.vmin), c.steep));
    const __m256 hi =
        basalt::utils::fast_sigmoid_avx(_mm256_mul_ps(_mm256_sub_ps(c.vmax, vel), c.steep));
    return _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(lo, hi));
}

/// Weight for 8 (f, k) points. `sym_mask` lanes (all bits set) are Nyquist
/// points whose weight is averaged over +v and -v; `any_sym` is a cheap
/// pre-check so the second evaluation is skipped for ordinary vectors.
/// k == 0 means infinite apparent velocity, which always passes (weight 1);
/// this includes DC (f == 0, k == 0).
BASALT_TARGET_AVX2 inline __m256 fk_weight_ps(__m256 vf, __m256 vk, __m256 sym_mask, bool any_sym,
                                              const FKBandConsts& c) {
    const __m256 vel = _mm256_div_ps(vf, vk);
    __m256 w = band_weight_ps(vel, c);
    if (any_sym) {
        const __m256 w_neg = band_weight_ps(_mm256_sub_ps(_mm256_setzero_ps(), vel), c);
        const __m256 avg = _mm256_mul_ps(_mm256_add_ps(w, w_neg), _mm256_set1_ps(0.5f));
        w = _mm256_blendv_ps(w, avg, sym_mask);
    }
    const __m256 k_is_zero = _mm256_cmp_ps(vk, _mm256_setzero_ps(), _CMP_EQ_OQ);
    return _mm256_blendv_ps(w, _mm256_set1_ps(1.0f), k_is_zero);
}

BASALT_TARGET_AVX2 inline __m256 absolute_ps(__m256 value) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return _mm256_and_ps(value, mask);
}

} // namespace basalt::filter::detail
