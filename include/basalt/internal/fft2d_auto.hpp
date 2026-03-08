#pragma once

#include "basalt/kernel/fft2d.hpp"

#include <cstddef>

namespace basalt::internal {

basalt::kernel::FFT2DConfig default_fft2d_config(size_t rows, size_t cols);
basalt::kernel::FFT2DConfig resolved_fft2d_config(const basalt::kernel::FFT2DConfig& config,
                                                  size_t rows, size_t cols);

} // namespace basalt::internal
