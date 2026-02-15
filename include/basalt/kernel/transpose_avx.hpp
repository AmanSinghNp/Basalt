#pragma once

#include <cstddef>

namespace basalt::kernel {

/// AVX2-accelerated 8×8 in-register transpose.
/// Transposes an 8×8 block from src (strided by src_stride) into
/// dst (strided by dst_stride). src and dst may point anywhere.
void transpose_8x8_avx(const float* src, float* dst,
                        size_t src_stride, size_t dst_stride);

/// AVX2-accelerated tiled transpose for full matrices.
/// Uses 8×8 AVX kernel for aligned blocks, scalar fallback for edges.
void transpose_tiled_avx(const float* src, float* dst,
                         size_t rows, size_t cols);

} // namespace basalt::kernel
