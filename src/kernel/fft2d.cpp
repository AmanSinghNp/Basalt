#include "basalt/kernel/fft2d.hpp"
#include "basalt/kernel/fft.hpp"
#include "basalt/kernel/transpose.hpp"
#include <cassert>
#include <iostream>
#include <cstdio>

namespace basalt::kernel {

namespace {

/// Apply 1D FFT to each row of a row-major rows×cols matrix (SoA).
/// Rows are contiguous, so this is cache-friendly.
void fft_rows(float* real, float* imag, 
              size_t rows, size_t cols, bool inverse) {
    for (size_t r = 0; r < rows; ++r) {
        float* row_real = real + r * cols;
        float* row_imag = imag + r * cols;

        if (inverse) {
            fft_inverse(row_real, row_imag, cols);
        } else {
            fft_forward(row_real, row_imag, cols);
        }
    }
}

void fft2d_impl(float* real, float* imag, 
                size_t rows, size_t cols, 
                basalt::MemoryArena& scratch, bool inverse) {
    assert(rows > 0 && (rows & (rows - 1)) == 0 && "rows must be power of 2");
    assert(cols > 0 && (cols & (cols - 1)) == 0 && "cols must be power of 2");

    size_t total = rows * cols;
    
    // Allocate scratch buffers for transpose
    float* tmp_real = scratch.allocate_array<float>(total);
    float* tmp_imag = scratch.allocate_array<float>(total);
    
    // Check allocation (though allocate_array usually returns valid or throws/asserts if logic matches)
    // Here we trust arena has space from caller check

    // --- Step 1: FFT along columns ---
    // Transpose: rows×cols → cols×rows (columns become rows)
    transpose_tiled(real, tmp_real, rows, cols);
    transpose_tiled(imag, tmp_imag, rows, cols);

    // Now tmp is cols×rows; each "row" was a column. FFT each row.
    fft_rows(tmp_real, tmp_imag, cols, rows, inverse);

    // Transpose back: cols×rows → rows×cols
    transpose_tiled(tmp_real, real, cols, rows);
    transpose_tiled(tmp_imag, imag, cols, rows);

    // --- Step 2: FFT along rows ---
    fft_rows(real, imag, rows, cols, inverse);
}

} // anonymous namespace

// FORCE REBUILD 1
void fft2d_forward(float* real, float* imag, 
                   size_t rows, size_t cols, 
                   basalt::MemoryArena& scratch) {
    fft2d_impl(real, imag, rows, cols, scratch, false);
}

void fft2d_inverse(float* real, float* imag, 
                   size_t rows, size_t cols, 
                   basalt::MemoryArena& scratch) {
    fft2d_impl(real, imag, rows, cols, scratch, true);
}

} // namespace basalt::kernel
