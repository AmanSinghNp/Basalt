#include "basalt/kernel/transpose.hpp"

#include <algorithm>

namespace basalt::kernel {

void transpose_tiled(const float* src, float* dst, 
                     size_t rows, size_t cols, 
                     size_t tile_size) {
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
