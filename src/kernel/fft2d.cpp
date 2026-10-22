#include "basalt/kernel/fft2d.hpp"
#include "basalt/internal/fft2d_auto.hpp"
#include "basalt/internal/fft_plan.hpp"
#include "basalt/kernel/transpose_avx.hpp"

#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>

namespace basalt::kernel {

namespace {

float* allocate_plane(basalt::MemoryArena& scratch, size_t total) {
    float* p = scratch.allocate_array<float>(total);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

void transpose_pair(const float* src_real, const float* src_imag, float* dst_real, float* dst_imag,
                    size_t rows, size_t cols, size_t tile) {
    transpose_tiled_avx(src_real, dst_real, rows, cols, tile);
    transpose_tiled_avx(src_imag, dst_imag, rows, cols, tile);
}

void fft2d_legacy_column_first(float* real, float* imag, size_t rows, size_t cols, float* tmp_real,
                               float* tmp_imag, bool inverse, float final_scale, size_t tile) {
    // Columns first (via transpose), then rows. The combined 1/(rows*cols)
    // scale is applied once, fused into the last 1D pass.
    transpose_pair(real, imag, tmp_real, tmp_imag, rows, cols, tile);
    internal::fft_rows_avx(tmp_real, tmp_imag, cols, rows, inverse, 1.0f);
    transpose_pair(tmp_real, tmp_imag, real, imag, cols, rows, tile);
    internal::fft_rows_avx(real, imag, rows, cols, inverse, final_scale);
}

void fft2d_four_step_row_first(float* real, float* imag, size_t rows, size_t cols, float* tmp_real,
                               float* tmp_imag, bool inverse, float final_scale, size_t tile) {
    internal::fft_rows_avx(real, imag, rows, cols, inverse, 1.0f);
    transpose_pair(real, imag, tmp_real, tmp_imag, rows, cols, tile);
    internal::fft_rows_avx(tmp_real, tmp_imag, cols, rows, inverse, final_scale);
    transpose_pair(tmp_real, tmp_imag, real, imag, cols, rows, tile);
}

void fft2d_row_first_column_strips(float* real, float* imag, size_t rows, size_t cols,
                                   basalt::MemoryArena& scratch, bool inverse, float final_scale) {
    // Rows in place, then the columns in strips (no transposes). The combined
    // scale is applied once, when the column pass writes its results back.
    // The strip buffers are taken first so bad_alloc leaves the data untouched.
    const size_t width = internal::fft_cols_strip_width(rows, cols);
    float* buf_real = allocate_plane(scratch, rows * width);
    float* buf_imag = allocate_plane(scratch, rows * width);
    internal::fft_rows_avx(real, imag, rows, cols, inverse, 1.0f);
    internal::fft_cols_strips(real, imag, rows, cols, inverse, final_scale, buf_real, buf_imag,
                              width);
}

bool is_pow2(size_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

void fft2d_impl(float* real, float* imag, size_t rows, size_t cols, basalt::MemoryArena& scratch,
                bool inverse, const basalt::kernel::FFT2DConfig& config, const char* who) {
    // --- Validate everything before touching any data (all build types) ---
    if (rows == 0 || cols == 0) {
        return; // Empty matrix: documented no-op.
    }
    if (!is_pow2(rows) || !is_pow2(cols)) {
        throw std::invalid_argument(std::string(who) + ": rows=" + std::to_string(rows) +
                                    " and cols=" + std::to_string(cols) +
                                    " must both be powers of two");
    }
    if (rows > internal::kMaxFFTSize || cols > internal::kMaxFFTSize || rows > SIZE_MAX / cols ||
        rows * cols > SIZE_MAX / sizeof(float)) {
        throw std::invalid_argument(std::string(who) + ": matrix is too large");
    }
    if (real == nullptr || imag == nullptr) {
        throw std::invalid_argument(std::string(who) + ": null data pointer");
    }
    if (real == imag) {
        throw std::invalid_argument(std::string(who) + ": real and imag must not alias");
    }

    const size_t total = rows * cols;
    const basalt::kernel::FFT2DConfig resolved =
        basalt::internal::resolved_fft2d_config(config, rows, cols);

    // --- Acquire scratch (throws std::bad_alloc, data still untouched) ---
    // Scratch is handed back on every exit path; caller allocations are kept.
    basalt::ArenaScope scope(scratch);

    // Single combined scale for the inverse (exact: both factors are powers of two).
    const float final_scale = inverse ? 1.0f / static_cast<float>(total) : 1.0f;
    const size_t tile = resolved.transpose_tile;

    switch (resolved.schedule) {
    case basalt::kernel::FFT2DSchedule::LegacyColumnFirst: {
        float* tmp_real = allocate_plane(scratch, total);
        float* tmp_imag = allocate_plane(scratch, total);
        fft2d_legacy_column_first(real, imag, rows, cols, tmp_real, tmp_imag, inverse, final_scale,
                                  tile);
        break;
    }
    case basalt::kernel::FFT2DSchedule::FourStepRowFirst: {
        float* tmp_real = allocate_plane(scratch, total);
        float* tmp_imag = allocate_plane(scratch, total);
        fft2d_four_step_row_first(real, imag, rows, cols, tmp_real, tmp_imag, inverse, final_scale,
                                  tile);
        break;
    }
    case basalt::kernel::FFT2DSchedule::RowFirstColumnStrips:
        fft2d_row_first_column_strips(real, imag, rows, cols, scratch, inverse, final_scale);
        break;
    case basalt::kernel::FFT2DSchedule::Auto:
        // resolved_fft2d_config never returns Auto; reject rather than skip silently.
        throw std::logic_error(std::string(who) + ": unresolved Auto schedule");
    }
}

} // namespace

void fft2d_forward(float* real, float* imag, size_t rows, size_t cols,
                   basalt::MemoryArena& scratch) {
    fft2d_forward(real, imag, rows, cols, scratch, FFT2DConfig{});
}

void fft2d_forward(float* real, float* imag, size_t rows, size_t cols, basalt::MemoryArena& scratch,
                   const FFT2DConfig& config) {
    fft2d_impl(real, imag, rows, cols, scratch, false, config, "fft2d_forward");
}

void fft2d_inverse(float* real, float* imag, size_t rows, size_t cols,
                   basalt::MemoryArena& scratch) {
    fft2d_inverse(real, imag, rows, cols, scratch, FFT2DConfig{});
}

void fft2d_inverse(float* real, float* imag, size_t rows, size_t cols, basalt::MemoryArena& scratch,
                   const FFT2DConfig& config) {
    fft2d_impl(real, imag, rows, cols, scratch, true, config, "fft2d_inverse");
}

} // namespace basalt::kernel
