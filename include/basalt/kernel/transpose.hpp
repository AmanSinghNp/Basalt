#pragma once

#include <cstddef>

namespace basalt::kernel {

/// Out-of-place cache-tiled matrix transpose.
/// Transposes a rows×cols matrix 'src' into a cols×rows matrix 'dst'.
/// 'tile_size' controls the cache tile (default 8 → 256 bytes fits L1);
/// a value of 0 is clamped to 1.
///
/// Preconditions (checked in every build type, std::invalid_argument on failure):
///   - 'src' and 'dst' are non-null and their rows*cols element ranges do not
///     overlap (in-place transpose is not supported, src == dst is rejected).
///   - rows*cols*sizeof(float) does not overflow size_t.
/// rows == 0 or cols == 0 is a no-op (pointers are not inspected).
void transpose_tiled(const float* src, float* dst, size_t rows, size_t cols, size_t tile_size = 8);

} // namespace basalt::kernel
