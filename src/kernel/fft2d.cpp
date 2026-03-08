#include "basalt/kernel/fft2d.hpp"
#include "basalt/internal/fft2d_auto.hpp"
#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/transpose_avx.hpp"

#include <cassert>

namespace basalt::kernel {

namespace {

/// Apply 1D FFT to each row of a row-major rows x cols matrix (SoA).
/// Rows are contiguous, so this is cache-friendly.
void fft_rows(float* real, float* imag,
              size_t rows, size_t cols, bool inverse) {
    for (size_t r = 0; r < rows; ++r) {
        float* row_real = real + r * cols;
        float* row_imag = imag + r * cols;

        if (inverse) {
            fft_inverse_avx(row_real, row_imag, cols);
        } else {
            fft_forward_avx(row_real, row_imag, cols);
        }
    }
}

void transpose_pair(const float* src_real, const float* src_imag,
                    float* dst_real, float* dst_imag,
                    size_t rows, size_t cols, size_t tile) {
    transpose_tiled_avx(src_real, dst_real, rows, cols, tile);
    transpose_tiled_avx(src_imag, dst_imag, rows, cols, tile);
}

void fft2d_legacy_column_first(float* real, float* imag,
                               size_t rows, size_t cols,
                               basalt::MemoryArena& scratch,
                               bool inverse, size_t tile) {
    const size_t total = rows * cols;
    float* tmp_real = scratch.allocate_array<float>(total);
    float* tmp_imag = scratch.allocate_array<float>(total);

    transpose_pair(real, imag, tmp_real, tmp_imag, rows, cols, tile);
    fft_rows(tmp_real, tmp_imag, cols, rows, inverse);
    transpose_pair(tmp_real, tmp_imag, real, imag, cols, rows, tile);
    fft_rows(real, imag, rows, cols, inverse);
}

void fft2d_four_step_row_first(float* real, float* imag,
                               size_t rows, size_t cols,
                               basalt::MemoryArena& scratch,
                               bool inverse, size_t tile) {
    assert(rows > 0 && (rows & (rows - 1)) == 0 && "rows must be power of 2");
    assert(cols > 0 && (cols & (cols - 1)) == 0 && "cols must be power of 2");

    const size_t total = rows * cols;
    float* tmp_real = scratch.allocate_array<float>(total);
    float* tmp_imag = scratch.allocate_array<float>(total);

    fft_rows(real, imag, rows, cols, inverse);
    transpose_pair(real, imag, tmp_real, tmp_imag, rows, cols, tile);
    fft_rows(tmp_real, tmp_imag, cols, rows, inverse);
    transpose_pair(tmp_real, tmp_imag, real, imag, cols, rows, tile);
}

void fft2d_impl(float* real, float* imag,
                size_t rows, size_t cols,
                basalt::MemoryArena& scratch, bool inverse,
                const basalt::kernel::FFT2DConfig& config) {
    assert(rows > 0 && (rows & (rows - 1)) == 0 && "rows must be power of 2");
    assert(cols > 0 && (cols & (cols - 1)) == 0 && "cols must be power of 2");

    const basalt::kernel::FFT2DConfig resolved = basalt::internal::resolved_fft2d_config(config, rows, cols);
    const size_t tile = resolved.transpose_tile;
    switch (resolved.schedule) {
    case basalt::kernel::FFT2DSchedule::LegacyColumnFirst:
        fft2d_legacy_column_first(real, imag, rows, cols, scratch, inverse, tile);
        break;
    case basalt::kernel::FFT2DSchedule::FourStepRowFirst:
        fft2d_four_step_row_first(real, imag, rows, cols, scratch, inverse, tile);
        break;
    case basalt::kernel::FFT2DSchedule::Auto:
        assert(false && "Auto schedule must be resolved before dispatch");
        break;
    }
}

} // namespace

void fft2d_forward(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch) {
    fft2d_forward(real, imag, rows, cols, scratch, FFT2DConfig{});
}

void fft2d_forward(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch,
                   const FFT2DConfig& config) {
    fft2d_impl(real, imag, rows, cols, scratch, false, config);
}

void fft2d_inverse(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch) {
    fft2d_inverse(real, imag, rows, cols, scratch, FFT2DConfig{});
}

void fft2d_inverse(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch,
                   const FFT2DConfig& config) {
    fft2d_impl(real, imag, rows, cols, scratch, true, config);
}

} // namespace basalt::kernel
