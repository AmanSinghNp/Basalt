#pragma once

#include <cstddef>

namespace basalt::kernel {

/// CPU dispatch (all functions in this header): the AVX code runs only when
/// cpu_features() reports AVX2 and FMA (and BASALT_SIMD_MAX does not cap it
/// below avx2); otherwise the call transparently runs a portable scalar
/// transpose with identical results. It never throws because of the CPU.
///
/// AVX2-accelerated 8x8 in-register transpose.
/// Transposes an 8x8 block from src (strided by src_stride) into
/// dst (strided by dst_stride). This is an unchecked hot-path primitive: the
/// caller guarantees that both 8x8 blocks are fully readable/writable and that
/// the source and destination blocks do not overlap.
void transpose_8x8_avx(const float* src, float* dst,
                       size_t src_stride, size_t dst_stride);

/// AVX2-accelerated tiled transpose for full matrices.
/// Uses a two-level tile:
///   1) cache tile (block_size x block_size, default 128; values below 8 are
///      raised to 8)
///   2) 8x8 AVX micro-kernel inside each cache tile.
/// Scalar fallback handles edge regions.
///
/// 'src' (rows×cols) and 'dst' (cols×rows) must not overlap; in-place use is
/// not supported. Checked in every build type: null pointers, overlapping
/// ranges (including src == dst) and rows*cols overflow throw
/// std::invalid_argument. rows == 0 or cols == 0 is a no-op.
void transpose_tiled_avx(const float* src, float* dst,
                         size_t rows, size_t cols,
                         size_t block_size = 128);

} // namespace basalt::kernel
