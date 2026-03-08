#pragma once

#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/simd_mode.hpp"

#include <immintrin.h>
#include <cstddef>

namespace basalt::filter::detail {

struct FKFilterExecutionPlan {
    size_t tile_rows = 64;
    size_t tile_cols = 128;
    size_t prefetch_distance = 16;
};

FKFilterExecutionPlan make_execution_plan(size_t rows, size_t cols);
bool cpu_supports_avx512_filter();

void apply_fk_filter_avx2(basalt::kernel::ComplexSoA& data,
                          size_t rows, size_t cols,
                          const FKParams& params,
                          const FKFilterExecutionPlan& plan);

void apply_fk_filter_avx512(basalt::kernel::ComplexSoA& data,
                            size_t rows, size_t cols,
                            const FKParams& params,
                            const FKFilterExecutionPlan& plan);

inline float wrapped_frequency(size_t index, size_t size, double step) {
    if (index <= size / 2) {
        return static_cast<float>(static_cast<double>(index) * step);
    }
    return static_cast<float>(
        static_cast<double>(static_cast<int>(index) - static_cast<int>(size)) * step
    );
}

inline float wrapped_wavenumber(size_t index, size_t size, double step) {
    if (index <= size / 2) {
        return static_cast<float>(static_cast<double>(index) * step);
    }
    return static_cast<float>(
        static_cast<double>(static_cast<int>(index) - static_cast<int>(size)) * step
    );
}

inline __m256 absolute_ps(__m256 value) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return _mm256_and_ps(value, mask);
}

} // namespace basalt::filter::detail
