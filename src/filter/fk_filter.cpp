#include "basalt/filter/fk_filter.hpp"
#include "basalt/utils/fast_math.hpp"

#include "fk_filter_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace basalt::filter {

namespace detail {

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

bool cpu_supports_avx512_filter() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

void apply_fk_filter_avx2(basalt::kernel::ComplexSoA& data,
                          size_t rows, size_t cols,
                          const FKParams& params,
                          const FKFilterExecutionPlan& plan) {
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    const float steepness = 10.0f / static_cast<float>(params.taper_width);
    const __m256 v_steep = _mm256_set1_ps(steepness);
    const __m256 v_min = _mm256_set1_ps(static_cast<float>(params.mute_vel_min));
    const __m256 v_max = _mm256_set1_ps(static_cast<float>(params.mute_vel_max));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 all_lanes = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    const __m256i lane_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // Even sizes have a Nyquist row (f = +f_N) and a Nyquist column (k = +k_N).
    const bool has_nyquist_col = (cols % 2) == 0;
    const size_t nyquist_col = cols / 2;

    for (size_t tile_row = 0; tile_row < rows; tile_row += plan.tile_rows) {
        const size_t row_end = std::min(tile_row + plan.tile_rows, rows);

        for (size_t tile_col = 0; tile_col < cols; tile_col += plan.tile_cols) {
            const size_t col_end = std::min(tile_col + plan.tile_cols, cols);

            for (size_t i = tile_row; i < row_end; ++i) {
                const __m256 vf = _mm256_set1_ps(detail::wrapped_frequency(i, rows, df));
                const bool row_is_nyquist = (rows % 2) == 0 && i == rows / 2;

                for (size_t j = tile_col; j < col_end; j += 8) {
                    const size_t next_prefetch = j + plan.prefetch_distance;
                    if (next_prefetch < cols) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.real + i * cols + next_prefetch),
                            _MM_HINT_T0
                        );
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.imag + i * cols + next_prefetch),
                            _MM_HINT_T0
                        );
                    } else if (i + 1 < row_end) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.real + (i + 1) * cols + tile_col),
                            _MM_HINT_T0
                        );
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.imag + (i + 1) * cols + tile_col),
                            _MM_HINT_T0
                        );
                    }

                    alignas(32) std::array<float, 8> k_vals{};
                    for (size_t lane = 0; lane < 8; ++lane) {
                        const size_t index = j + lane;
                        k_vals[lane] = index < cols
                            ? detail::wrapped_wavenumber(index, cols, dk)
                            : 1.0f;
                    }

                    const __m256 vk = _mm256_load_ps(k_vals.data());
                    // Lanes with k == 0 (zero-wavenumber column, incl. DC) have v = +/-inf and
                    // are kept. Their divisor is replaced by 1 so the division stays finite,
                    // and their weight is forced to 1 by the blend below.
                    const __m256 k_is_zero = _mm256_cmp_ps(vk, zero, _CMP_EQ_OQ);
                    const __m256 vk_safe = _mm256_blendv_ps(vk, one, k_is_zero);
                    const __m256 vel = _mm256_div_ps(vf, vk_safe);
                    // Signed band [v_min, v_max] with a sigmoid edge on each side.
                    const __m256 rise = basalt::utils::fast_sigmoid_avx(
                        _mm256_mul_ps(_mm256_sub_ps(vel, v_min), v_steep));
                    const __m256 fall = basalt::utils::fast_sigmoid_avx(
                        _mm256_mul_ps(_mm256_sub_ps(v_max, vel), v_steep));
                    __m256 inside = _mm256_mul_ps(rise, fall);
                    // Nyquist bins are their own alias of both signs (f_N = -f_N, k_N = -k_N),
                    // so their velocity is ambiguous. They use the mean of the weights at v and -v,
                    // which makes w(f, k) == w(-f, -k) hold for every bin and any band.
                    const bool nyquist_col_here = has_nyquist_col && nyquist_col >= j &&
                                                  nyquist_col - j < 8;
                    if (row_is_nyquist || nyquist_col_here) {
                        const __m256 mirror_vel = _mm256_sub_ps(zero, vel);
                        const __m256 mirror_rise = basalt::utils::fast_sigmoid_avx(
                            _mm256_mul_ps(_mm256_sub_ps(mirror_vel, v_min), v_steep));
                        const __m256 mirror_fall = basalt::utils::fast_sigmoid_avx(
                            _mm256_mul_ps(_mm256_sub_ps(v_max, mirror_vel), v_steep));
                        const __m256 mirror_inside = _mm256_mul_ps(mirror_rise, mirror_fall);
                        const __m256 mean_inside = _mm256_mul_ps(
                            half, _mm256_add_ps(inside, mirror_inside));
                        const __m256 nyquist_lanes = row_is_nyquist
                            ? all_lanes
                            : _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                                  lane_index, _mm256_set1_epi32(static_cast<int>(nyquist_col - j))));
                        inside = _mm256_blendv_ps(inside, mean_inside, nyquist_lanes);
                    }
                    const __m256 mute_weight = _mm256_sub_ps(one, inside);
                    const __m256 weight = _mm256_blendv_ps(mute_weight, one, k_is_zero);

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

namespace {

// Double-precision helpers for the out-of-range path. Index `index` of an axis with `size`
// bins and step `step` is wrapped to the signed range, as in the single-precision helpers.
double wrapped_axis_double(size_t index, size_t size, double step) {
    if (index <= size / 2) {
        return static_cast<double>(index) * step;
    }
    return -static_cast<double>(size - index) * step;
}

// Logistic sigmoid 1 / (1 + exp(-distance * steepness)). A NaN argument only arises from
// 0 * inf (an exact tie with an infinite steepness), where the limit of the sigmoid is 1/2.
double edge_sigmoid(double distance, double steepness) {
    double arg = distance * steepness;
    if (std::isnan(arg)) {
        arg = 0.0;
    }
    return 1.0 / (1.0 + std::exp(-arg));
}

} // namespace

bool float_path_is_in_range(const FKParams& params, size_t rows, size_t cols) {
    constexpr double kFloatMax = static_cast<double>(std::numeric_limits<float>::max());
    constexpr double kFloatMin = static_cast<double>(std::numeric_limits<float>::min());
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);
    const double steepness = 10.0 / params.taper_width;
    const auto normal = [](double x) { return x >= kFloatMin && x <= kFloatMax; };
    return normal(steepness) && normal(df) && normal(dk) &&
           static_cast<double>(rows / 2) * df <= kFloatMax &&
           static_cast<double>(cols / 2) * dk <= kFloatMax &&
           std::abs(params.mute_vel_min) <= kFloatMax &&
           std::abs(params.mute_vel_max) <= kFloatMax;
}

void apply_fk_filter_double(basalt::kernel::ComplexSoA& data,
                            size_t rows, size_t cols,
                            const FKParams& params) {
    constexpr double kMax = std::numeric_limits<double>::max();
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);
    const double steepness = 10.0 / params.taper_width;
    const bool has_nyquist_col = (cols % 2) == 0;
    const size_t nyquist_col = cols / 2;

    const auto inside_at = [&](double velocity) {
        const double rise = edge_sigmoid(velocity - params.mute_vel_min, steepness);
        const double fall = edge_sigmoid(params.mute_vel_max - velocity, steepness);
        return rise * fall;
    };

    for (size_t i = 0; i < rows; ++i) {
        const double f = wrapped_axis_double(i, rows, df);
        const bool row_is_nyquist = (rows % 2) == 0 && i == rows / 2;
        for (size_t j = 0; j < cols; ++j) {
            const double k = wrapped_axis_double(j, cols, dk);
            if (k == 0.0) {
                continue; // zero-wavenumber column (incl. DC) is always kept
            }
            // Clamping keeps the velocity finite; a saturated sigmoid is unchanged by it.
            const double velocity = std::clamp(f / k, -kMax, kMax);
            double inside = inside_at(velocity);
            if (row_is_nyquist || (has_nyquist_col && j == nyquist_col)) {
                inside = 0.5 * (inside + inside_at(-velocity));
            }
            const double weight = 1.0 - inside;
            const size_t offset = i * cols + j;
            data.real[offset] = static_cast<float>(static_cast<double>(data.real[offset]) * weight);
            data.imag[offset] = static_cast<float>(static_cast<double>(data.imag[offset]) * weight);
        }
    }
}

} // namespace detail

void apply_fk_filter(basalt::kernel::ComplexSoA& data,
                     size_t rows, size_t cols,
                     const FKParams& params) {
    apply_fk_filter(data, rows, cols, params, basalt::kernel::SimdMode::Auto);
}

void apply_fk_filter(basalt::kernel::ComplexSoA& data,
                     size_t rows, size_t cols,
                     const FKParams& params,
                     basalt::kernel::SimdMode simd_mode) {
    // No-op for a non-positive taper, an empty band (min >= max, also catches NaN) or empty data.
    if (!(params.taper_width > 0.0) || !(params.mute_vel_min < params.mute_vel_max) ||
        rows == 0 || cols == 0) {
        return;
    }

    // The SIMD paths compute in float and are only exact while every float quantity is normal
    // and finite. Outside that range the scalar double-precision path is used for any simd_mode.
    if (!detail::float_path_is_in_range(params, rows, cols)) {
        detail::apply_fk_filter_double(data, rows, cols, params);
        return;
    }

    const detail::FKFilterExecutionPlan plan = detail::make_execution_plan(rows, cols);
    const bool can_use_avx512 = simd_mode != basalt::kernel::SimdMode::AVX2 &&
                                detail::cpu_supports_avx512_filter();

    if (simd_mode == basalt::kernel::SimdMode::AVX512 && !can_use_avx512) {
        detail::apply_fk_filter_avx2(data, rows, cols, params, plan);
        return;
    }

    if ((simd_mode == basalt::kernel::SimdMode::AVX512 ||
         (simd_mode == basalt::kernel::SimdMode::Auto && can_use_avx512))) {
        detail::apply_fk_filter_avx512(data, rows, cols, params, plan);
        return;
    }

    detail::apply_fk_filter_avx2(data, rows, cols, params, plan);
}

} // namespace basalt::filter
