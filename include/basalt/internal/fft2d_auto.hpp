#pragma once

#include "basalt/kernel/fft2d.hpp"

#include <cstddef>

namespace basalt::internal {

/// Schedule/tile choice for a rows x cols transform. Every shape currently
/// gets RowFirstColumnStrips (measured fastest at all sizes, see fft2d_auto.cpp);
/// the parameters stay so size classes can return if a future schedule needs them.
basalt::kernel::FFT2DConfig default_fft2d_config(size_t rows, size_t cols);

/// Returns 'config' unchanged unless its schedule is Auto, in which case the
/// result is default_fft2d_config(rows, cols). Never returns Auto.
basalt::kernel::FFT2DConfig resolved_fft2d_config(const basalt::kernel::FFT2DConfig& config,
                                                  size_t rows, size_t cols);

} // namespace basalt::internal
