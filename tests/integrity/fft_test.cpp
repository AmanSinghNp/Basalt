#include "basalt/kernel/fft.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/arena.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace basalt::kernel;

// ---------- Round-trip (Parseval) tests ----------

class FFTRoundTripTest : public ::testing::TestWithParam<size_t> {};

TEST_P(FFTRoundTripTest, ForwardInversePreservesData) {
    size_t n = GetParam();
    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    // Fill with known pattern: x[i] = sin(2π·3·i/n) + 0.5·cos(2π·7·i/n)
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n);
        data.real[i] = static_cast<float>(std::sin(2.0 * M_PI * 3.0 * t) +
                                          0.5 * std::cos(2.0 * M_PI * 7.0 * t));
        data.imag[i] = 0.0f;
    }

    // Save original
    std::vector<float> original(data.real, data.real + n);

    // Forward then inverse
    fft_forward(data.real, data.imag, n);
    fft_inverse(data.real, data.imag, n);

    // Check reconstruction
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(data.real[i], original[i], 1e-4f)
            << "Mismatch at index " << i << " for n=" << n;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-4f)
            << "Imaginary part nonzero at index " << i << " for n=" << n;
    }
}

INSTANTIATE_TEST_SUITE_P(
    FFTSizes, FFTRoundTripTest,
    ::testing::Values(8, 16, 64, 256, 1024, 4096)
);

// ---------- Impulse response test ----------

TEST(FFTTest, ImpulseResponse) {
    constexpr size_t n = 64;
    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    // Delta function: x[0] = 1, rest = 0
    data.clear();
    data.real[0] = 1.0f;

    fft_forward(data.real, data.imag, n);

    // FFT of delta should be flat: all real parts = 1.0, all imag parts = 0.0
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(data.real[i], 1.0f, 1e-5f)
            << "Real part not 1.0 at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-5f)
            << "Imag part not 0.0 at bin " << i;
    }
}

// ---------- Known sinusoid test ----------

TEST(FFTTest, SingleSinusoidPeak) {
    constexpr size_t n = 256;
    constexpr size_t freq_bin = 10;  // Frequency we'll inject

    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);
    data.clear();

    // Generate pure sinusoid at bin 'freq_bin'
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n);
        data.real[i] = static_cast<float>(std::cos(2.0 * M_PI * freq_bin * t));
    }

    fft_forward(data.real, data.imag, n);

    // Compute magnitudes
    float max_mag = 0.0f;
    size_t max_bin = 0;
    for (size_t i = 0; i < n; ++i) {
        float mag = std::sqrt(data.real[i] * data.real[i] + 
                              data.imag[i] * data.imag[i]);
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = i;
        }
    }

    // Peak should be at freq_bin (or conjugate at n - freq_bin)
    EXPECT_TRUE(max_bin == freq_bin || max_bin == n - freq_bin)
        << "Peak at bin " << max_bin << ", expected " << freq_bin;

    // Magnitude at peak should be n/2 (for a real cosine signal)
    EXPECT_NEAR(max_mag, static_cast<float>(n) / 2.0f, 1.0f);
}

// ---------- DC signal test ----------

TEST(FFTTest, DCSignal) {
    constexpr size_t n = 64;
    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    // All ones (DC)
    for (size_t i = 0; i < n; ++i) {
        data.real[i] = 1.0f;
        data.imag[i] = 0.0f;
    }

    fft_forward(data.real, data.imag, n);

    // DC bin should be n, all others 0
    EXPECT_NEAR(data.real[0], static_cast<float>(n), 1e-4f);
    EXPECT_NEAR(data.imag[0], 0.0f, 1e-4f);

    for (size_t i = 1; i < n; ++i) {
        EXPECT_NEAR(data.real[i], 0.0f, 1e-4f) << "Nonzero real at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-4f) << "Nonzero imag at bin " << i;
    }
}

// ---------- Energy preservation (Parseval's theorem) ----------

TEST(FFTTest, EnergyPreservation) {
    constexpr size_t n = 512;
    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    // Random-ish signal
    for (size_t i = 0; i < n; ++i) {
        data.real[i] = static_cast<float>(std::sin(0.1 * static_cast<double>(i)) * std::cos(0.03 * static_cast<double>(i)));
        data.imag[i] = 0.0f;
    }

    // Time-domain energy
    double time_energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        time_energy += static_cast<double>(data.real[i]) * data.real[i];
    }

    fft_forward(data.real, data.imag, n);

    // Frequency-domain energy (Parseval: sum |X[k]|^2 / N = sum |x[n]|^2)
    double freq_energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        freq_energy += static_cast<double>(data.real[i]) * data.real[i] +
                       static_cast<double>(data.imag[i]) * data.imag[i];
    }
    freq_energy /= static_cast<double>(n);

    EXPECT_NEAR(freq_energy, time_energy, time_energy * 1e-4)
        << "Parseval's theorem violated";
}
