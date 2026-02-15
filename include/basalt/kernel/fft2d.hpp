#pragma once

#include "basalt/arena.hpp"
#include <cstddef>

namespace basalt::kernel {

/// In-place 2D forward FFT.
/// Data is stored as flat row-major arrays of size rows*cols.
/// 'real' and 'imag' are SoA arrays.
/// 'scratch' arena provides temporary storage for transpose buffers.
/// Both 'rows' and 'cols' MUST be powers of 2.
void fft2d_forward(float* real, float* imag, 
                   size_t rows, size_t cols, 
                   basalt::MemoryArena& scratch);

/// In-place 2D inverse FFT with 1/(rows*cols) scaling.
/// Both 'rows' and 'cols' MUST be powers of 2.
void fft2d_inverse(float* real, float* imag, 
                   size_t rows, size_t cols, 
                   basalt::MemoryArena& scratch);

} // namespace basalt::kernel
