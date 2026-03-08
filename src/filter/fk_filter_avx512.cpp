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
    const __m512 v_cut = _mm512_set1_ps(static_cast<float>(params.mute_vel_max));
    const __m512 v_eps = _mm512_set1_ps(1e-6f);
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

    for (size_t tile_row = 0; tile_row < rows; tile_row += plan.tile_rows) {
        const size_t row_end = std::min(tile_row + plan.tile_rows, rows);

        for (size_t tile_col = 0; tile_col < cols; tile_col += plan.tile_cols) {
            const size_t col_end = std::min(tile_col + plan.tile_cols, cols);

            for (size_t i = tile_row; i < row_end; ++i) {
                const __m512 vf = _mm512_set1_ps(wrapped_frequency(i, rows, df));

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
                    const __m512 vk = _mm512_maskz_load_ps(lane_mask, k_vals.data());
                    const __m512 vel = _mm512_div_ps(vf, _mm512_add_ps(vk, v_eps));
                    __m512 x = _mm512_sub_ps(_mm512_and_ps(vel, abs_mask), v_cut);
                    x = _mm512_mul_ps(x, v_steep);
                    const __m512 weight = _mm512_mask_mov_ps(
                        _mm512_setzero_ps(),
                        lane_mask,
                        fast_sigmoid_avx512_split(x)
                    );
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
