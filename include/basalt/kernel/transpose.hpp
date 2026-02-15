#pragma once

#include <cstddef>

namespace basalt::kernel {

/// Out-of-place cache-tiled matrix transpose.
/// Transposes a rows×cols matrix 'src' into a cols×rows matrix 'dst'.
/// 'tile_size' controls the cache tile (default 8 → 256 bytes fits L1).
/// 'src' and 'dst' must not overlap.
void transpose_tiled(const float* src, float* dst, 
                     size_t rows, size_t cols, 
                     size_t tile_size = 8);

} // namespace basalt::kernel
