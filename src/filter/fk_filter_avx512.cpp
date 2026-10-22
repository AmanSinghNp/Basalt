#include "fk_filter_impl.hpp"

#include "basalt/utils/fast_math.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <immintrin.h>

namespace basalt::filter::detail {

namespace {

// Needs AVX-512F (zmm, masks) and AVX-512DQ (_mm512_insertf32x8) only; the VL
// and BW subsets are not used, so they are not required.
BASALT_TARGET_AVX512
void apply_fk_filter_avx512_impl(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                                 const FKParams& params, const FKFilterExecutionPlan& plan) {
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    const FKBandConsts band = make_band_consts(params);

    for (size_t tile_row = 0; tile_row < rows; tile_row += plan.tile_rows) {
        const size_t row_end = std::min(tile_row + plan.tile_rows, rows);

        for (size_t tile_col = 0; tile_col < cols; tile_col += plan.tile_cols) {
            const size_t col_end = std::min(tile_col + plan.tile_cols, cols);

            for (size_t i = tile_row; i < row_end; ++i) {
                const float f_val = wrapped_frequency(i, rows, df);
                const __m256 vf = _mm256_set1_ps(f_val);
                const bool row_nyquist = is_nyquist_index(i, rows);

                for (size_t j = tile_col; j < col_end; j += 16) {
                    const size_t next_prefetch = j + plan.prefetch_distance;
                    if (next_prefetch < cols) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.real + i * cols + next_prefetch),
                            _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(data.imag + i * cols + next_prefetch),
                            _MM_HINT_T0);
                    }

                    alignas(64) std::array<float, 16> k_vals{};
                    alignas(64) std::array<uint32_t, 16> sym_vals{};
                    bool any_sym = false;
                    for (size_t lane = 0; lane < 16; ++lane) {
                        const size_t index = j + lane;
                        k_vals[lane] = index < cols ? wrapped_wavenumber(index, cols, dk) : 1.0f;
                        const bool sym =
                            index < cols && (row_nyquist || is_nyquist_index(index, cols));
                        sym_vals[lane] = sym ? 0xFFFFFFFFu : 0u;
                        any_sym = any_sym || sym;
                    }

                    const size_t remaining = col_end - j;
                    const __mmask16 lane_mask =
                        remaining >= 16 ? static_cast<__mmask16>(0xFFFF)
                                        : static_cast<__mmask16>((1u << remaining) - 1u);

                    // The weight math is shared with the AVX2 path and runs on
                    // two 256-bit halves.
                    const __m256 w_lo =
                        fk_weight_ps(vf, _mm256_load_ps(k_vals.data()),
                                     _mm256_castsi256_ps(_mm256_load_si256(
                                         reinterpret_cast<const __m256i*>(sym_vals.data()))),
                                     any_sym, band);
                    const __m256 w_hi =
                        fk_weight_ps(vf, _mm256_load_ps(k_vals.data() + 8),
                                     _mm256_castsi256_ps(_mm256_load_si256(
                                         reinterpret_cast<const __m256i*>(sym_vals.data() + 8))),
                                     any_sym, band);
                    const __m512 weight = _mm512_mask_mov_ps(
                        _mm512_setzero_ps(), lane_mask,
                        _mm512_insertf32x8(_mm512_castps256_ps512(w_lo), w_hi, 1));
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

void apply_fk_filter_avx512(basalt::kernel::ComplexSoA& data, size_t rows, size_t cols,
                            const FKParams& params, const FKFilterExecutionPlan& plan) {
    apply_fk_filter_avx512_impl(data, rows, cols, params, plan);
}

} // namespace basalt::filter::detail
