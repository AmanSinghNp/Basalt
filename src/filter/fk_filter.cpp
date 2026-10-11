#include "basalt/filter/fk_filter.hpp"
#include "basalt/utils/fast_math.hpp"

#include "fk_filter_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>

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
    const __m256 zero = _mm256_setzero_ps();

    for (size_t tile_row = 0; tile_row < rows; tile_row += plan.tile_rows) {
        const size_t row_end = std::min(tile_row + plan.tile_rows, rows);

        for (size_t tile_col = 0; tile_col < cols; tile_col += plan.tile_cols) {
            const size_t col_end = std::min(tile_col + plan.tile_cols, cols);

            for (size_t i = tile_row; i < row_end; ++i) {
                const __m256 vf = _mm256_set1_ps(detail::wrapped_frequency(i, rows, df));

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
                    const __m256 inside = _mm256_mul_ps(rise, fall);
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
