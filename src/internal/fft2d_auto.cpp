#include "basalt/internal/fft2d_auto.hpp"

namespace basalt::internal {

basalt::kernel::FFT2DConfig default_fft2d_config(size_t rows, size_t cols) {
    const size_t max_dim = rows > cols ? rows : cols;

    // Exact sizes with individually tuned choices (unchanged).
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

    // Size classes for everything else (rectangular and other square sizes).
    // A matrix with at least as many elements as 1024x1024 does not fit in
    // cache and has long power-of-two strides; the 128-wide tile measured
    // clearly slower than 96 on very elongated shapes (e.g. 8192x512), so such
    // matrices share the tuned 1024x1024 choice. Anything smaller keeps the
    // 128-wide tile of the 512x512 choice.
    constexpr size_t kLargeElements = size_t{1} << 20;
    const bool large = cols != 0 && rows >= (kLargeElements + cols - 1) / cols;
    if (large) {
        return {basalt::kernel::FFT2DSchedule::LegacyColumnFirst, 96};
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
