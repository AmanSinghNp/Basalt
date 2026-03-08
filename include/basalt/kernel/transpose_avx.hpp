#pragma once

#include <cstddef>

namespace basalt::kernel {

/// AVX2-accelerated 8x8 in-register transpose.
/// Transposes an 8x8 block from src (strided by src_stride) into
/// dst (strided by dst_stride). src and dst may point anywhere.
void transpose_8x8_avx(const float* src, float* dst,
                       size_t src_stride, size_t dst_stride);

/// AVX2-accelerated tiled transpose for full matrices.
/// Uses a two-level tile:
///   1) cache tile (block_size x block_size, default 128)
///   2) 8x8 AVX micro-kernel inside each cache tile.
/// Scalar fallback handles edge regions.
void transpose_tiled_avx(const float* src, float* dst,
                         size_t rows, size_t cols,
                         size_t block_size = 128);

} // namespace basalt::kernel
