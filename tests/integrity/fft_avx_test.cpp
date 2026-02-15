#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/fft.hpp"
#include "basalt/kernel/cpu_state.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/transpose_avx.hpp"
#include "basalt/kernel/transpose.hpp"
#include "basalt/arena.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace basalt::kernel;

// ---------- FTZ/DAZ test ----------

TEST(CPUStateTest, FlushToZeroWorks) {
    set_flush_to_zero();
    EXPECT_TRUE(check_ftz_daz());
}

// ---------- AVX FFT Cross-validation against scalar ----------

class FFTAvxCrossValidation : public ::testing::TestWithParam<size_t> {};

TEST_P(FFTAvxCrossValidation, MatchesScalarForward) {
    size_t n = GetParam();

    // Two identical copies of data
    basalt::MemoryArena arena1(n * 2 * sizeof(float) + 4096);
    basalt::MemoryArena arena2(n * 2 * sizeof(float) + 4096);
    ComplexSoA scalar_data(arena1, n);
    ComplexSoA avx_data(arena2, n);

    // Fill with same pattern
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n);
        float val = static_cast<float>(
            std::sin(2.0 * M_PI * 5.0 * t) + 0.3 * std::cos(2.0 * M_PI * 13.0 * t));
        scalar_data.real[i] = val;
        scalar_data.imag[i] = 0.0f;
        avx_data.real[i] = val;
        avx_data.imag[i] = 0.0f;
    }

    fft_forward(scalar_data.real, scalar_data.imag, n);
    fft_forward_avx(avx_data.real, avx_data.imag, n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(avx_data.real[i], scalar_data.real[i], 1e-4f)
            << "Real mismatch at bin " << i << " for n=" << n;
        EXPECT_NEAR(avx_data.imag[i], scalar_data.imag[i], 1e-4f)
            << "Imag mismatch at bin " << i << " for n=" << n;
    }
}

TEST_P(FFTAvxCrossValidation, RoundTripPreservesData) {
    size_t n = GetParam();

    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n);
        data.real[i] = static_cast<float>(
            std::sin(2.0 * M_PI * 3.0 * t) + 0.5 * std::cos(2.0 * M_PI * 7.0 * t));
        data.imag[i] = 0.0f;
    }

    std::vector<float> original(data.real, data.real + n);

    fft_forward_avx(data.real, data.imag, n);
    fft_inverse_avx(data.real, data.imag, n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(data.real[i], original[i], 1e-4f)
            << "Mismatch at index " << i << " for n=" << n;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-4f)
            << "Imaginary nonzero at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    AvxFFTSizes, FFTAvxCrossValidation,
    ::testing::Values(8, 16, 32, 64, 256, 1024, 4096)
);

// ---------- AVX Transpose cross-validation ----------

TEST(TransposeAvxTest, MatchesScalar) {
    constexpr size_t rows = 64, cols = 128;
    std::vector<float> src(rows * cols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(i) * 0.01f;
    }

    std::vector<float> dst_scalar(rows * cols);
    std::vector<float> dst_avx(rows * cols);

    transpose_tiled(src.data(), dst_scalar.data(), rows, cols);
    transpose_tiled_avx(src.data(), dst_avx.data(), rows, cols);

    for (size_t i = 0; i < dst_scalar.size(); ++i) {
        EXPECT_FLOAT_EQ(dst_avx[i], dst_scalar[i])
            << "Mismatch at flat index " << i;
    }
}

TEST(TransposeAvxTest, RoundTrip) {
    constexpr size_t rows = 128, cols = 64;
    std::vector<float> original(rows * cols);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<float>(i);
    }

    std::vector<float> transposed(rows * cols);
    std::vector<float> restored(rows * cols);

    transpose_tiled_avx(original.data(), transposed.data(), rows, cols);
    transpose_tiled_avx(transposed.data(), restored.data(), cols, rows);

    EXPECT_EQ(original, restored);
}

TEST(TransposeAvxTest, NonAlignedEdges) {
    // 13×19 — neither dimension divisible by 8
    constexpr size_t rows = 13, cols = 19;
    std::vector<float> src(rows * cols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(i);
    }

    std::vector<float> dst_scalar(rows * cols);
    std::vector<float> dst_avx(rows * cols);

    transpose_tiled(src.data(), dst_scalar.data(), rows, cols);
    transpose_tiled_avx(src.data(), dst_avx.data(), rows, cols);

    for (size_t i = 0; i < dst_scalar.size(); ++i) {
        EXPECT_FLOAT_EQ(dst_avx[i], dst_scalar[i])
            << "Mismatch at index " << i << " for 13x19 matrix";
    }
}

// ---------- AVX FFT Impulse ----------

TEST(FFTAvxTest, ImpulseResponse) {
    constexpr size_t n = 64;
    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);
    data.clear();
    data.real[0] = 1.0f;

    fft_forward_avx(data.real, data.imag, n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(data.real[i], 1.0f, 1e-5f);
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-5f);
    }
}

// ---------- AVX FFT Energy preservation ----------

TEST(FFTAvxTest, EnergyPreservation) {
    constexpr size_t n = 512;
    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    for (size_t i = 0; i < n; ++i) {
        data.real[i] = static_cast<float>(
            std::sin(0.1 * static_cast<double>(i)) * 
            std::cos(0.03 * static_cast<double>(i)));
        data.imag[i] = 0.0f;
    }

    double time_energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        time_energy += static_cast<double>(data.real[i]) * data.real[i];
    }

    fft_forward_avx(data.real, data.imag, n);

    double freq_energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        freq_energy += static_cast<double>(data.real[i]) * data.real[i] +
                       static_cast<double>(data.imag[i]) * data.imag[i];
    }
    freq_energy /= static_cast<double>(n);

    EXPECT_NEAR(freq_energy, time_energy, time_energy * 1e-4);
}
