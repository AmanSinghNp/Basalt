#include "basalt/filter/fk_filter.hpp"
#include "basalt/utils/fast_math.hpp"

#include "basalt/internal/simd_dispatch.hpp"

#include "fk_filter_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace basalt::filter {

namespace detail {

void validate_params(const FKParams& params) {
    if (!(std::isfinite(params.dt) && params.dt > 0.0)) {
        throw std::invalid_argument("FKParams: dt must be finite and > 0");
    }
    if (!(std::isfinite(params.dx) && params.dx > 0.0)) {
        throw std::invalid_argument("FKParams: dx must be finite and > 0");
    }
    if (!(std::isfinite(params.taper_width) && params.taper_width > 0.0)) {
        throw std::invalid_argument("FKParams: taper_width must be finite and > 0");
    }
    if (!std::isfinite(params.mute_vel_min) || !std::isfinite(params.mute_vel_max)) {
        throw std::invalid_argument("FKParams: mute velocities must be finite");
    }
    if (!(params.mute_vel_min < params.mute_vel_max)) {
        throw std::invalid_argument("FKParams: mute_vel_min must be < mute_vel_max");
    }
}

void validate_buffer(const basalt::kernel::ComplexSoA& data, size_t rows, size_t cols) {
    if (cols > std::numeric_limits<size_t>::max() / rows) {
        throw std::invalid_argument("apply_fk_filter: rows * cols overflows");
    }
    if (!data.real || !data.imag) {
        throw std::invalid_argument("apply_fk_filter: data buffers are null");
    }
    if (rows * cols > data.count) {
        throw std::invalid_argument("apply_fk_filter: rows * cols exceeds data.count");
    }
}

FKFilterExecutionPlan make_execution_plan(size_t rows, size_t cols) {
    FKFilterExecutionPlan plan;
    if (rows <= 256 && cols <= 256) {
        plan.tile_rows = 32;
        plan.tile_cols = 64;
        plan.prefetch_distance = 8;
    } else if (rows >= 2048 || cols >= 2048) {
        plan.tile_rows = 32;
        plan.tile_cols = 96;
        plan.prefetch_distance = 24;
    }
    return plan;
}

namespace {

/// Scalar twin of fast_sigmoid_avx: 1 / (1 + exp(-x)), input clamped to
/// [-87, 87], NaN propagates. Uses std::exp rather than the AVX polynomial.
float sigmoid_scalar(float x) {
    if (x > 87.0f) {
        x = 87.0f;
    } else if (x < -87.0f) {
        x = -87.0f;
    }
    return 1.0f / (1.0f + std::exp(-x));
}

/// Scalar twin of band_weight_ps: 1 - S((v-min)s) * S((max-v)s).
float band_weight_scalar(float vel, float steep, float vmin, float vmax) {
    const float lo = sigmoid_scalar((vel - vmin) * steep);
    const float hi = sigmoid_scalar((vmax - vel) * steep);
    return 1.0f - lo * hi;
}

} // namespace

void apply_fk_filter_scalar(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                            const FKParams& params) {
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    // Same single-precision constants as make_band_consts.
    const float steep = 10.0f / static_cast<float>(params.taper_width);
    const float vmin = static_cast<float>(params.mute_vel_min);
    const float vmax = static_cast<float>(params.mute_vel_max);

    for (size_t i = 0; i < rows; ++i) {
        const float f = wrapped_frequency(i, rows, df);
        const bool row_nyquist = is_nyquist_index(i, rows);
        float* row_real = data.real + i * cols;
        float* row_imag = data.imag + i * cols;

        for (size_t j = 0; j < cols; ++j) {
            const float k = wrapped_wavenumber(j, cols, dk);
            float w;
            if (k == 0.0f) {
                w = 1.0f; // infinite apparent velocity (includes DC) always passes
            } else {
                const float vel = f / k;
                w = band_weight_scalar(vel, steep, vmin, vmax);
                if (row_nyquist || is_nyquist_index(j, cols)) {
                    // Nyquist has no defined sign: average the weight over +v and -v.
                    const float w_neg = band_weight_scalar(-vel, steep, vmin, vmax);
                    w = (w + w_neg) * 0.5f;
                }
            }
            row_real[j] *= w;
            row_imag[j] *= w;
        }
    }
}

BASALT_TARGET_AVX2
void apply_fk_filter_avx2(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                          const FKParams& params, const FKFilterExecutionPlan& plan) {
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    const detail::FKBandConsts band = detail::make_band_consts(params);

    for (size_t tile_row = 0; tile_row < rows; tile_row += plan.tile_rows) {
        const size_t row_end = std::min(tile_row + plan.tile_rows, rows);

        for (size_t tile_col = 0; tile_col < cols; tile_col += plan.tile_cols) {
            const size_t col_end = std::min(tile_col + plan.tile_cols, cols);

            for (size_t i = tile_row; i < row_end; ++i) {
                const __m256 vf = _mm256_set1_ps(detail::wrapped_frequency(i, rows, df));
                const bool row_nyquist = detail::is_nyquist_index(i, rows);

                for (size_t j = tile_col; j < col_end; j += 8) {
                    const size_t next_prefetch = j + plan.prefetch_distance;
                    if (next_prefetch < cols) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.real + i * cols + next_prefetch),
                            _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.imag + i * cols + next_prefetch),
                            _MM_HINT_T0);
                    } else if (i + 1 < row_end) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.real + (i + 1) * cols + tile_col),
                            _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.imag + (i + 1) * cols + tile_col),
                            _MM_HINT_T0);
                    }

                    alignas(32) std::array<float, 8> k_vals{};
                    alignas(32) std::array<uint32_t, 8> sym_vals{};
                    bool any_sym = false;
                    for (size_t lane = 0; lane < 8; ++lane) {
                        const size_t index = j + lane;
                        k_vals[lane] =
                            index < cols ? detail::wrapped_wavenumber(index, cols, dk) : 1.0f;
                        const bool sym =
                            index < cols && (row_nyquist || detail::is_nyquist_index(index, cols));
                        sym_vals[lane] = sym ? 0xFFFFFFFFu : 0u;
                        any_sym = any_sym || sym;
                    }

                    const __m256 vk = _mm256_load_ps(k_vals.data());
                    const __m256 sym_mask = _mm256_castsi256_ps(
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(sym_vals.data())));
                    const __m256 weight = detail::fk_weight_ps(vf, vk, sym_mask, any_sym, band);

                    const size_t remaining = col_end - j;
                    if (remaining >= 8) {
                        const size_t offset = i * cols + j;
                        __m256 real = _mm256_loadu_ps(data.real + offset);
                        __m256 imag = _mm256_loadu_ps(data.imag + offset);
                        real = _mm256_mul_ps(real, weight);
                        imag = _mm256_mul_ps(imag, weight);
                        _mm256_storeu_ps(data.real + offset, real);
                        _mm256_storeu_ps(data.imag + offset, imag);
                    } else {
                        alignas(32) std::array<float, 8> weights{};
                        _mm256_store_ps(weights.data(), weight);
                        for (size_t lane = 0; lane < remaining; ++lane) {
                            const size_t offset = i * cols + j + lane;
                            data.real[offset] *= weights[lane];
                            data.imag[offset] *= weights[lane];
                        }
                    }
                }
            }
        }
    }
}

} // namespace detail

void apply_fk_filter(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                     const FKParams& params) {
    apply_fk_filter(data, rows, cols, params, basalt::kernel::SimdMode::Auto);
}

void apply_fk_filter(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                     const FKParams& params, basalt::kernel::SimdMode simd_mode) {
    detail::validate_params(params);
    if (rows == 0 || cols == 0) {
        return;
    }
    detail::validate_buffer(data, rows, cols);

    // One dispatch decision per call. Requests are best-effort: an explicit
    // mode is clamped to what the CPU (and BASALT_SIMD_MAX) allows.
    switch (basalt::kernel::detail::resolve_simd_level(simd_mode)) {
    case basalt::kernel::detail::SimdLevel::AVX512:
        detail::apply_fk_filter_avx512(data, rows, cols, params,
                                       detail::make_execution_plan(rows, cols));
        break;
    case basalt::kernel::detail::SimdLevel::AVX2:
        detail::apply_fk_filter_avx2(data, rows, cols, params,
                                     detail::make_execution_plan(rows, cols));
        break;
    case basalt::kernel::detail::SimdLevel::Scalar:
        detail::apply_fk_filter_scalar(data, rows, cols, params);
        break;
    }
}

} // namespace basalt::filter
