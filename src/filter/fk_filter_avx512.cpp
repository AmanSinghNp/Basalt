#include "fk_filter_impl.hpp"

#include "basalt/utils/fast_math.hpp"

#include <algorithm>
#include <array>
#include <immintrin.h>

namespace basalt::filter::detail {

namespace {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512vl,avx512bw,avx512dq")))
#endif
__m512 fast_sigmoid_avx512_split(__m512 x) {
    const __m256 lo = _mm512_castps512_ps256(x);
    const __m256 hi = _mm512_extractf32x8_ps(x, 1);
    const __m256 lo_sigmoid = basalt::utils::fast_sigmoid_avx(lo);
    const __m256 hi_sigmoid = basalt::utils::fast_sigmoid_avx(hi);

    __m512 result = _mm512_castps256_ps512(lo_sigmoid);
    result = _mm512_insertf32x8(result, hi_sigmoid, 1);
    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512vl,avx512bw,avx512dq")))
#endif
void apply_fk_filter_avx512_impl(basalt::kernel::ComplexSoA& data,
                                 size_t rows, size_t cols,
                                 const FKParams& params,
                                 const FKFilterExecutionPlan& plan) {
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    const __m512 v_steep = _mm512_set1_ps(10.0f / static_cast<float>(params.taper_width));
    const __m512 v_min = _mm512_set1_ps(static_cast<float>(params.mute_vel_min));
    const __m512 v_max = _mm512_set1_ps(static_cast<float>(params.mute_vel_max));
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 zero = _mm512_setzero_ps();
    // Even sizes have a Nyquist row (f = +f_N) and a Nyquist column (k = +k_N).
    const bool has_nyquist_col = (cols % 2) == 0;
    const size_t nyquist_col = cols / 2;

    for (size_t tile_row = 0; tile_row < rows; tile_row += plan.tile_rows) {
        const size_t row_end = std::min(tile_row + plan.tile_rows, rows);

        for (size_t tile_col = 0; tile_col < cols; tile_col += plan.tile_cols) {
            const size_t col_end = std::min(tile_col + plan.tile_cols, cols);

            for (size_t i = tile_row; i < row_end; ++i) {
                const __m512 vf = _mm512_set1_ps(wrapped_frequency(i, rows, df));
                const bool row_is_nyquist = (rows % 2) == 0 && i == rows / 2;

                for (size_t j = tile_col; j < col_end; j += 16) {
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
                    }

                    alignas(64) std::array<float, 16> k_vals{};
                    for (size_t lane = 0; lane < 16; ++lane) {
                        const size_t index = j + lane;
                        k_vals[lane] = index < cols
                            ? wrapped_wavenumber(index, cols, dk)
                            : 1.0f;
                    }

                    const size_t remaining = col_end - j;
                    const __mmask16 lane_mask = remaining >= 16
                        ? static_cast<__mmask16>(0xFFFF)
                        : static_cast<__mmask16>((1u << remaining) - 1u);
                    // k_vals holds all 16 lanes (1.0 past the last column), so a full load is valid.
                    const __m512 vk = _mm512_load_ps(k_vals.data());
                    // k == 0 lanes (zero-wavenumber column, incl. DC) have v = +/-inf and are kept:
                    // the divisor is replaced by 1 and the weight is forced to 1 via the mask.
                    const __mmask16 k_is_zero = _mm512_cmp_ps_mask(vk, zero, _CMP_EQ_OQ);
                    const __m512 vk_safe = _mm512_mask_mov_ps(vk, k_is_zero, one);
                    const __m512 vel = _mm512_div_ps(vf, vk_safe);
                    // Signed band [v_min, v_max] with a sigmoid edge on each side.
                    const __m512 rise = fast_sigmoid_avx512_split(
                        _mm512_mul_ps(_mm512_sub_ps(vel, v_min), v_steep));
                    const __m512 fall = fast_sigmoid_avx512_split(
                        _mm512_mul_ps(_mm512_sub_ps(v_max, vel), v_steep));
                    __m512 inside = _mm512_mul_ps(rise, fall);
                    // Nyquist bins use the mean of the weights at v and -v (see fk_filter.cpp), so
                    // that w(f, k) == w(-f, -k) holds for every bin and any band.
                    const bool nyquist_col_here = has_nyquist_col && nyquist_col >= j &&
                                                  nyquist_col - j < 16;
                    if (row_is_nyquist || nyquist_col_here) {
                        const __m512 mirror_vel = _mm512_sub_ps(zero, vel);
                        const __m512 mirror_rise = fast_sigmoid_avx512_split(
                            _mm512_mul_ps(_mm512_sub_ps(mirror_vel, v_min), v_steep));
                        const __m512 mirror_fall = fast_sigmoid_avx512_split(
                            _mm512_mul_ps(_mm512_sub_ps(v_max, mirror_vel), v_steep));
                        const __m512 mirror_inside = _mm512_mul_ps(mirror_rise, mirror_fall);
                        const __m512 mean_inside = _mm512_mul_ps(
                            half, _mm512_add_ps(inside, mirror_inside));
                        const __mmask16 nyquist_lanes = row_is_nyquist
                            ? static_cast<__mmask16>(0xFFFF)
                            : static_cast<__mmask16>(1u << static_cast<unsigned>(nyquist_col - j));
                        inside = _mm512_mask_mov_ps(inside, nyquist_lanes, mean_inside);
                    }
                    const __m512 mute_weight = _mm512_sub_ps(one, inside);
                    const __m512 weight = _mm512_mask_mov_ps(mute_weight, k_is_zero, one);
                    const size_t offset = i * cols + j;
                    __m512 real = _mm512_maskz_loadu_ps(lane_mask, data.real + offset);
                    __m512 imag = _mm512_maskz_loadu_ps(lane_mask, data.imag + offset);
                    real = _mm512_mul_ps(real, weight);
                    imag = _mm512_mul_ps(imag, weight);
                    _mm512_mask_storeu_ps(data.real + offset, lane_mask, real);
                    _mm512_mask_storeu_ps(data.imag + offset, lane_mask, imag);
                }
            }
        }
    }
}

} // namespace

void apply_fk_filter_avx512(basalt::kernel::ComplexSoA& data,
                            size_t rows, size_t cols,
                            const FKParams& params,
                            const FKFilterExecutionPlan& plan) {
#if defined(__GNUC__) || defined(__clang__)
    apply_fk_filter_avx512_impl(data, rows, cols, params, plan);
#else
    (void)data;
    (void)rows;
    (void)cols;
    (void)params;
    (void)plan;
#endif
}

} // namespace basalt::filter::detail
