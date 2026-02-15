#include "basalt/kernel/transpose_avx.hpp"
#include "basalt/kernel/transpose.hpp"

#include <immintrin.h>
#include <algorithm>

namespace basalt::kernel {

void transpose_8x8_avx(const float* src, float* dst,
                        size_t src_stride, size_t dst_stride) {
    // Load 8 rows of 8 floats each
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    // Stage 1: Interleave 32-bit elements
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);  // r0[0],r1[0],r0[1],r1[1], r0[4],r1[4],r0[5],r1[5]
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);  // r0[2],r1[2],r0[3],r1[3], r0[6],r1[6],r0[7],r1[7]
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    // Stage 2: Interleave 64-bit elements
    __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);  // 0b01_00_01_00
    __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);  // 0b11_10_11_10
    __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    // Stage 3: Permute 128-bit lanes  
    __m256 v0 = _mm256_permute2f128_ps(u0, u4, 0x20);  // low lanes
    __m256 v1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    __m256 v2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    __m256 v3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    __m256 v4 = _mm256_permute2f128_ps(u0, u4, 0x31);  // high lanes
    __m256 v5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    __m256 v6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    __m256 v7 = _mm256_permute2f128_ps(u3, u7, 0x31);

    // Store transposed rows
    _mm256_storeu_ps(dst + 0 * dst_stride, v0);
    _mm256_storeu_ps(dst + 1 * dst_stride, v1);
    _mm256_storeu_ps(dst + 2 * dst_stride, v2);
    _mm256_storeu_ps(dst + 3 * dst_stride, v3);
    _mm256_storeu_ps(dst + 4 * dst_stride, v4);
    _mm256_storeu_ps(dst + 5 * dst_stride, v5);
    _mm256_storeu_ps(dst + 6 * dst_stride, v6);
    _mm256_storeu_ps(dst + 7 * dst_stride, v7);
}

void transpose_tiled_avx(const float* src, float* dst,
                         size_t rows, size_t cols) {
    // Process full 8×8 tiles with AVX2
    size_t r8 = (rows / 8) * 8;
    size_t c8 = (cols / 8) * 8;

    for (size_t bi = 0; bi < r8; bi += 8) {
        for (size_t bj = 0; bj < c8; bj += 8) {
            // Transpose 8×8 block at (bi, bj) in src → (bj, bi) in dst
            transpose_8x8_avx(
                src + bi * cols + bj,      // src pointer
                dst + bj * rows + bi,      // dst pointer
                cols,                       // src stride (next row in src)
                rows                        // dst stride (next row in dst)
            );
        }
    }

    // Handle remaining columns (right edge, cols not divisible by 8)
    if (c8 < cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = c8; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    // Handle remaining rows (bottom edge, rows not divisible by 8)
    if (r8 < rows) {
        for (size_t i = r8; i < rows; ++i) {
            for (size_t j = 0; j < c8; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
}

} // namespace basalt::kernel
