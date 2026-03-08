#include "basalt/kernel/fft2d.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/arena.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace basalt::kernel;

// ---------- 2D Round-trip test ----------

class FFT2DRoundTripTest : public ::testing::TestWithParam<std::pair<size_t, size_t>> {};

TEST_P(FFT2DRoundTripTest, ForwardInversePreservesData) {
    auto [rows, cols] = GetParam();
    size_t total = rows * cols;

    // Data arena + generous scratch arena
    basalt::MemoryArena data_arena(total * 2 * sizeof(float) + 4096);
    basalt::MemoryArena scratch(total * 2 * sizeof(float) + 4096);

    ComplexSoA data(data_arena, total);

    // Fill with a 2D pattern
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            double x = static_cast<double>(c) / cols;
            double y = static_cast<double>(r) / rows;
            data.real[r * cols + c] = static_cast<float>(
                std::sin(2.0 * M_PI * 3.0 * x) * std::cos(2.0 * M_PI * 2.0 * y));
            data.imag[r * cols + c] = 0.0f;
        }
    }

    // Save original
    std::vector<float> original(data.real, data.real + total);

    // Forward then inverse
    fft2d_forward(data.real, data.imag, rows, cols, scratch);
    
    // Reset scratch for inverse (arena is linear, not reusable without reset)
    scratch.reset();
    
    fft2d_inverse(data.real, data.imag, rows, cols, scratch);

    // Check reconstruction
    for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(data.real[i], original[i], 1e-3f)
            << "Real mismatch at flat index " << i 
            << " for " << rows << "x" << cols;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-3f)
            << "Imag nonzero at flat index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    FFT2DSizes, FFT2DRoundTripTest,
    ::testing::Values(
        std::make_pair(8, 8),
        std::make_pair(16, 16),
        std::make_pair(16, 32),
        std::make_pair(32, 16),
        std::make_pair(64, 64)
    )
);

// ---------- 2D Impulse response ----------

TEST(FFT2DTest, ImpulseResponse) {
    constexpr size_t rows = 8, cols = 8;
    constexpr size_t total = rows * cols;

    basalt::MemoryArena data_arena(total * 2 * sizeof(float) + 4096);
    basalt::MemoryArena scratch(total * 2 * sizeof(float) + 4096);

    ComplexSoA data(data_arena, total);
    data.clear();
    data.real[0] = 1.0f;  // 2D delta

    fft2d_forward(data.real, data.imag, rows, cols, scratch);

    // 2D FFT of delta should be flat: all bins = 1.0 + 0.0i
    for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(data.real[i], 1.0f, 1e-4f)
            << "Real not 1.0 at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-4f)
            << "Imag not 0.0 at bin " << i;
    }
}

// ---------- 2D DC signal ----------

TEST(FFT2DTest, DCSignal) {
    constexpr size_t rows = 16, cols = 16;
    constexpr size_t total = rows * cols;

    basalt::MemoryArena data_arena(total * 2 * sizeof(float) + 4096);
    basalt::MemoryArena scratch(total * 2 * sizeof(float) + 4096);

    ComplexSoA data(data_arena, total);
    for (size_t i = 0; i < total; ++i) {
        data.real[i] = 1.0f;
        data.imag[i] = 0.0f;
    }

    fft2d_forward(data.real, data.imag, rows, cols, scratch);

    // DC bin (0,0) should be rows*cols = 256
    EXPECT_NEAR(data.real[0], static_cast<float>(total), 1e-2f);
    EXPECT_NEAR(data.imag[0], 0.0f, 1e-4f);

    // All other bins should be ~0
    for (size_t i = 1; i < total; ++i) {
        EXPECT_NEAR(data.real[i], 0.0f, 1e-3f)
            << "Nonzero real at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-3f)
            << "Nonzero imag at bin " << i;
    }
}

// ---------- Energy preservation (2D Parseval's) ----------

TEST(FFT2DTest, EnergyPreservation) {
    constexpr size_t rows = 32, cols = 32;
    constexpr size_t total = rows * cols;

    basalt::MemoryArena data_arena(total * 2 * sizeof(float) + 4096);
    basalt::MemoryArena scratch(total * 2 * sizeof(float) + 4096);

    ComplexSoA data(data_arena, total);
    for (size_t i = 0; i < total; ++i) {
        data.real[i] = static_cast<float>(std::sin(0.1 * static_cast<double>(i)) * std::cos(0.03 * static_cast<double>(i)));
        data.imag[i] = 0.0f;
    }

    // Time-domain energy
    double time_energy = 0.0;
    for (size_t i = 0; i < total; ++i) {
        time_energy += static_cast<double>(data.real[i]) * data.real[i];
    }

    fft2d_forward(data.real, data.imag, rows, cols, scratch);

    // Frequency-domain energy: sum|X|^2 / N
    double freq_energy = 0.0;
    for (size_t i = 0; i < total; ++i) {
        freq_energy += static_cast<double>(data.real[i]) * data.real[i] +
                       static_cast<double>(data.imag[i]) * data.imag[i];
    }
    freq_energy /= static_cast<double>(total);

    EXPECT_NEAR(freq_energy, time_energy, time_energy * 1e-3)
        << "2D Parseval's theorem violated";
}

TEST(FFT2DTest, ScheduleVariantsMatch) {
    constexpr size_t rows = 32;
    constexpr size_t cols = 64;
    constexpr size_t total = rows * cols;
    constexpr size_t arena_bytes = total * 2 * sizeof(float) + 4096;

    std::vector<float> input_real(total);
    std::vector<float> input_imag(total, 0.0f);
    for (size_t i = 0; i < total; ++i) {
        input_real[i] = static_cast<float>(
            std::sin(0.07 * static_cast<double>(i)) +
            0.25 * std::cos(0.19 * static_cast<double>(i))
        );
    }

    std::vector<float> legacy_real = input_real;
    std::vector<float> legacy_imag = input_imag;
    std::vector<float> four_real = input_real;
    std::vector<float> four_imag = input_imag;
    std::vector<float> auto_real = input_real;
    std::vector<float> auto_imag = input_imag;

    {
        basalt::MemoryArena scratch(arena_bytes);
        fft2d_forward(
            legacy_real.data(), legacy_imag.data(), rows, cols, scratch,
            FFT2DConfig{FFT2DSchedule::LegacyColumnFirst, 128}
        );
    }
    {
        basalt::MemoryArena scratch(arena_bytes);
        fft2d_forward(
            four_real.data(), four_imag.data(), rows, cols, scratch,
            FFT2DConfig{FFT2DSchedule::FourStepRowFirst, 128}
        );
    }
    {
        basalt::MemoryArena scratch(arena_bytes);
        fft2d_forward(auto_real.data(), auto_imag.data(), rows, cols, scratch);
    }

    for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(legacy_real[i], four_real[i], 1e-3f) << "Real mismatch at " << i;
        EXPECT_NEAR(legacy_imag[i], four_imag[i], 1e-3f) << "Imag mismatch at " << i;
        EXPECT_NEAR(auto_real[i], legacy_real[i], 1e-6f) << "Auto real mismatch at " << i;
        EXPECT_NEAR(auto_imag[i], legacy_imag[i], 1e-6f) << "Auto imag mismatch at " << i;
    }
}
