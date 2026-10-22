#include "basalt/internal/fft2d_auto.hpp"

namespace basalt::internal {

basalt::kernel::FFT2DConfig default_fft2d_config(size_t rows, size_t cols) {
    // The transpose-free column-strip schedule measured faster than both
    // transpose schedules at every size and shape tried (16x16 up to
    // 4096x1024 / 1024x4096, forward and inverse, AVX2 and scalar): it reads
    // and writes the matrix once for the column pass instead of making four
    // transposing passes over it. The transpose schedules stay available as
    // explicit choices. The tile only matters for those, so the default 128 is
    // kept for the field.
    (void)rows;
    (void)cols;
    return {basalt::kernel::FFT2DSchedule::RowFirstColumnStrips, 128};
}

basalt::kernel::FFT2DConfig resolved_fft2d_config(const basalt::kernel::FFT2DConfig& config,
                                                  size_t rows, size_t cols) {
    if (config.schedule != basalt::kernel::FFT2DSchedule::Auto) {
        return config;
    }
    return default_fft2d_config(rows, cols);
}

} // namespace basalt::internal
