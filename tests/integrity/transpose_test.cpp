#include "basalt/kernel/transpose.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>
#include <numeric>

using namespace basalt::kernel;

TEST(TransposeTest, SmallKnownMatrix) {
    // 3x4 matrix → 4x3
    //  1  2  3  4
    //  5  6  7  8
    //  9 10 11 12
    std::vector<float> src = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    std::vector<float> dst(12);

    transpose_tiled(src.data(), dst.data(), 3, 4);

    // Expected 4x3 result:
    //  1  5  9
    //  2  6 10
    //  3  7 11
    //  4  8 12
    std::vector<float> expected = {1,5,9, 2,6,10, 3,7,11, 4,8,12};
    EXPECT_EQ(dst, expected);
}

TEST(TransposeTest, SquareMatrix) {
    // 4x4 identity-like pattern
    constexpr size_t n = 4;
    std::vector<float> src(n * n);
    std::iota(src.begin(), src.end(), 1.0f);
    
    std::vector<float> dst(n * n);
    transpose_tiled(src.data(), dst.data(), n, n);

    // Verify A^T[j][i] == A[i][j]
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(dst[j * n + i], src[i * n + j])
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST(TransposeTest, RoundTrip) {
    constexpr size_t rows = 64;
    constexpr size_t cols = 128;
    
    std::vector<float> original(rows * cols);
    std::iota(original.begin(), original.end(), 0.0f);

    std::vector<float> transposed(rows * cols);
    std::vector<float> restored(rows * cols);

    // Transpose: rows×cols → cols×rows
    transpose_tiled(original.data(), transposed.data(), rows, cols);
    // Transpose back: cols×rows → rows×cols
    transpose_tiled(transposed.data(), restored.data(), cols, rows);

    EXPECT_EQ(original, restored);
}

TEST(TransposeTest, SingleRow) {
    // 1×8 → 8×1
    std::vector<float> src = {1,2,3,4,5,6,7,8};
    std::vector<float> dst(8);

    transpose_tiled(src.data(), dst.data(), 1, 8);
    EXPECT_EQ(src, dst);  // 1×N transpose is identity in memory layout
}

TEST(TransposeTest, SingleColumn) {
    // 8×1 → 1×8
    std::vector<float> src = {1,2,3,4,5,6,7,8};
    std::vector<float> dst(8);

    transpose_tiled(src.data(), dst.data(), 8, 1);
    EXPECT_EQ(src, dst);
}

TEST(TransposeTest, NonPowerOfTwo) {
    // 5×7 matrix
    constexpr size_t rows = 5, cols = 7;
    std::vector<float> src(rows * cols);
    std::iota(src.begin(), src.end(), 1.0f);

    std::vector<float> dst(rows * cols);
    transpose_tiled(src.data(), dst.data(), rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            EXPECT_FLOAT_EQ(dst[j * rows + i], src[i * cols + j]);
        }
    }
}

TEST(TransposeTest, LargeRectangular) {
    constexpr size_t rows = 100, cols = 200;
    std::vector<float> src(rows * cols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(i) * 0.01f;
    }

    std::vector<float> dst(rows * cols);
    std::vector<float> restored(rows * cols);

    transpose_tiled(src.data(), dst.data(), rows, cols);
    transpose_tiled(dst.data(), restored.data(), cols, rows);

    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_FLOAT_EQ(restored[i], src[i]) << "at flat index " << i;
    }
}

// ---------- Validation / robustness ----------

TEST(TransposeValidationTest, ZeroTileSizeDoesNotHang) {
    // tile_size == 0 used to loop forever (bi += 0).
    constexpr size_t rows = 5, cols = 7;
    std::vector<float> src(rows * cols);
    std::iota(src.begin(), src.end(), 1.0f);
    std::vector<float> dst(rows * cols, 0.0f);

    transpose_tiled(src.data(), dst.data(), rows, cols, 0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            EXPECT_FLOAT_EQ(dst[j * rows + i], src[i * cols + j]);
        }
    }
}

TEST(TransposeValidationTest, HugeTileSizeIsClamped) {
    constexpr size_t rows = 9, cols = 4;
    std::vector<float> src(rows * cols);
    std::iota(src.begin(), src.end(), 1.0f);
    std::vector<float> expected(rows * cols);
    transpose_tiled(src.data(), expected.data(), rows, cols, 1);

    std::vector<float> dst(rows * cols, 0.0f);
    transpose_tiled(src.data(), dst.data(), rows, cols, std::numeric_limits<size_t>::max());
    EXPECT_EQ(dst, expected);
}

TEST(TransposeValidationTest, NullPointersThrow) {
    std::vector<float> buf(16, 0.0f);
    EXPECT_THROW(transpose_tiled(nullptr, buf.data(), 4, 4), std::invalid_argument);
    EXPECT_THROW(transpose_tiled(buf.data(), nullptr, 4, 4), std::invalid_argument);
}

TEST(TransposeValidationTest, OverlappingBuffersThrow) {
    std::vector<float> buf(64, 0.0f);
    EXPECT_THROW(transpose_tiled(buf.data(), buf.data(), 4, 4), std::invalid_argument);
    EXPECT_THROW(transpose_tiled(buf.data(), buf.data() + 8, 4, 4), std::invalid_argument);
    EXPECT_THROW(transpose_tiled(buf.data() + 8, buf.data(), 4, 4), std::invalid_argument);
    // Adjacent ranges do not overlap.
    EXPECT_NO_THROW(transpose_tiled(buf.data(), buf.data() + 16, 4, 4));
}

TEST(TransposeValidationTest, EmptyMatrixIsNoOpAndOverflowThrows) {
    EXPECT_NO_THROW(transpose_tiled(nullptr, nullptr, 0, 5));
    EXPECT_NO_THROW(transpose_tiled(nullptr, nullptr, 5, 0));
    float x = 0.0f, y = 0.0f;
    EXPECT_THROW(transpose_tiled(&x, &y, std::numeric_limits<size_t>::max() / 2, 4),
                 std::invalid_argument);
}
