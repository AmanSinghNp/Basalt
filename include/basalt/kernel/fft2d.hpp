#pragma once

#include "basalt/arena.hpp"
#include <cstddef>
#include <limits>

namespace basalt::kernel {

enum class FFT2DSchedule {
    Auto,
    LegacyColumnFirst,
    FourStepRowFirst,
    /// Row FFTs in place, then column FFTs without transposes: strips of
    /// adjacent columns are gathered into a small contiguous buffer and
    /// transformed with the SIMD lanes running across columns. 'transpose_tile'
    /// is ignored. Needs far less scratch than the transpose schedules.
    RowFirstColumnStrips
};

struct FFT2DConfig {
    FFT2DSchedule schedule = FFT2DSchedule::Auto;
    size_t transpose_tile = 128;
};

/// Upper bound on the scratch arena bytes one fft2d_forward / fft2d_inverse
/// call can need for a rows x cols transform: two float planes of rows*cols
/// elements, each with worst-case alignment slack and MemoryArena's trailing
/// alias padding. The transpose schedules need exactly this much; the
/// RowFirstColumnStrips schedule needs at most this (two buffers of rows * min(cols, 64)
/// floats), so the bound holds for every schedule. 'config' is part of the
/// signature so future schedules can differ. Returns 0 for an empty
/// matrix and SIZE_MAX if the size does not fit in size_t.
/// Size the scratch arena with at least this many bytes (on top of anything
/// else the caller allocates from the same arena).
constexpr size_t fft2d_scratch_bytes(size_t rows, size_t cols,
                                     const FFT2DConfig& config = FFT2DConfig{}) noexcept {
    (void)config;
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    if (rows == 0 || cols == 0) {
        return 0;
    }
    if (cols > kMax / rows) {
        return kMax;
    }
    const size_t total = rows * cols;
    constexpr size_t kSlack = basalt::MemoryArena::kAliasPadding + 63;  // padding + max align fix-up
    if (total > (kMax / 2 - kSlack) / sizeof(float)) {
        return kMax;
    }
    return 2 * (total * sizeof(float) + kSlack);
}

/// Sign convention, scaling and size rules
/// ----------------------------------------
/// All transforms are in-place on flat row-major arrays of size rows*cols,
/// stored as separate SoA 'real' and 'imag' arrays (distinct, non-null).
/// Forward uses exp(-2*pi*i*(k*r/rows + l*c/cols)) (see fft.hpp); inverse is
/// the conjugate transform scaled once by 1/(rows*cols).
///
/// 'rows' and 'cols' must each be a power of two; otherwise (or for null/aliasing
/// pointers) std::invalid_argument is thrown before any data is modified.
/// rows == 0 or cols == 0 is a no-op.
///
/// Scratch memory: every call takes two rows*cols float planes from 'scratch'
/// (see fft2d_scratch_bytes). If the arena cannot supply them, std::bad_alloc
/// is thrown before any data is modified. The scratch is taken inside an
/// ArenaScope, so the arena is rewound to its entry state when the call returns
/// or throws: allocations the caller made before the call stay valid, and
/// repeated calls on the same arena need no manual reset().

/// In-place 2D forward FFT with the Auto schedule.
void fft2d_forward(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch);

/// In-place 2D forward FFT with explicit schedule/tile selection.
void fft2d_forward(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch,
                   const FFT2DConfig& config);

/// In-place 2D inverse FFT with 1/(rows*cols) scaling (applied once).
void fft2d_inverse(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch);

/// In-place 2D inverse FFT with explicit schedule/tile selection.
void fft2d_inverse(float* real, float* imag,
                   size_t rows, size_t cols,
                   basalt::MemoryArena& scratch,
                   const FFT2DConfig& config);

} // namespace basalt::kernel
