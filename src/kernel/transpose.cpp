#include "basalt/kernel/transpose.hpp"
#include "basalt/internal/fft_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace basalt::internal {

bool validate_transpose_args(const float* src, const float* dst, size_t rows, size_t cols,
                             const char* who) {
    if (rows == 0 || cols == 0) {
        return false; // Empty matrix: documented no-op, pointers not inspected.
    }
    if (rows > SIZE_MAX / cols || rows * cols > SIZE_MAX / sizeof(float)) {
        throw std::invalid_argument(std::string(who) + ": rows*cols overflows");
    }
    if (src == nullptr || dst == nullptr) {
        throw std::invalid_argument(std::string(who) + ": null pointer");
    }
    const size_t bytes = rows * cols * sizeof(float);
    const uintptr_t s = reinterpret_cast<uintptr_t>(src);
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
    // Half-open byte ranges [s, s+bytes) and [d, d+bytes) must be disjoint.
    if (s < d + bytes && d < s + bytes) {
        throw std::invalid_argument(
            std::string(who) +
            ": src and dst ranges overlap (in-place transpose is not supported)");
    }
    return true;
}

} // namespace basalt::internal

namespace basalt::kernel {

void transpose_tiled(const float* src, float* dst, size_t rows, size_t cols, size_t tile_size) {
    if (!internal::validate_transpose_args(src, dst, rows, cols, "transpose_tiled")) {
        return;
    }
    // A zero tile would never advance the loops below; clamp to 1.
    if (tile_size == 0) {
        tile_size = 1;
    }
    // A tile larger than the matrix is equivalent to one tile; clamping also
    // keeps bi + tile_size from wrapping around for absurd tile sizes.
    tile_size = std::min(tile_size, std::max(rows, cols));

    // Process full tiles
    for (size_t bi = 0; bi < rows; bi += tile_size) {
        for (size_t bj = 0; bj < cols; bj += tile_size) {
            // Tile boundaries (handle edge tiles)
            size_t i_end = std::min(bi + tile_size, rows);
            size_t j_end = std::min(bj + tile_size, cols);

            // Transpose within the tile
            for (size_t i = bi; i < i_end; ++i) {
                for (size_t j = bj; j < j_end; ++j) {
                    // src[i][j] → dst[j][i]
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        }
    }
}

} // namespace basalt::kernel
