#include "basalt/internal/fft2d_auto.hpp"

namespace basalt::internal {

basalt::kernel::FFT2DConfig default_fft2d_config(size_t rows, size_t cols) {
    const size_t max_dim = rows > cols ? rows : cols;
    if (rows == 256 && cols == 256) {
        return {basalt::kernel::FFT2DSchedule::FourStepRowFirst, 128};
    }
    if (rows == 512 && cols == 512) {
        return {basalt::kernel::FFT2DSchedule::LegacyColumnFirst, 128};
    }
    if (rows == 1024 && cols == 1024) {
        return {basalt::kernel::FFT2DSchedule::LegacyColumnFirst, 96};
    }
    if (max_dim >= 2048 && rows == cols) {
        return {basalt::kernel::FFT2DSchedule::FourStepRowFirst, 96};
    }
    return {basalt::kernel::FFT2DSchedule::LegacyColumnFirst, 128};
}

basalt::kernel::FFT2DConfig resolved_fft2d_config(const basalt::kernel::FFT2DConfig& config,
                                                  size_t rows, size_t cols) {
    if (config.schedule != basalt::kernel::FFT2DSchedule::Auto) {
        return config;
    }
    return default_fft2d_config(rows, cols);
}

} // namespace basalt::internal
