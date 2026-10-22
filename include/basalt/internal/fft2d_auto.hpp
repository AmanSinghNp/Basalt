#pragma once

#include "basalt/kernel/fft2d.hpp"

#include <cstddef>

namespace basalt::internal {

/// Tuned schedule/tile choice for a rows x cols transform. Exact tuned entries
/// exist for 256x256, 512x512, 1024x1024 and square sizes >= 2048; every other
/// shape is chosen by size class (total element count), so rectangular
/// matrices such as 4096x1024 or 8192x512 get a sensible config too.
basalt::kernel::FFT2DConfig default_fft2d_config(size_t rows, size_t cols);

/// Returns 'config' unchanged unless its schedule is Auto, in which case the
/// result is default_fft2d_config(rows, cols). Never returns Auto.
basalt::kernel::FFT2DConfig resolved_fft2d_config(const basalt::kernel::FFT2DConfig& config,
                                                  size_t rows, size_t cols);

} // namespace basalt::internal
