#include "basalt/arena.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/fft.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace basalt::kernel;

namespace {

// ---------- Double-precision reference DFT (O(n^2), definition-level) ----------

/// Forward (sign = -1) or inverse-direction unscaled (sign = +1) DFT in double.
void reference_dft(const std::vector<float>& in_re, const std::vector<float>& in_im,
                   std::vector<double>& out_re, std::vector<double>& out_im, int sign) {
    const size_t n = in_re.size();
    // Twiddle table exp(sign * 2*pi*i*m/n), reused via (k*j) mod n.
    std::vector<double> cs(n), sn(n);
    for (size_t m = 0; m < n; ++m) {
        const double a = sign * 2.0 * M_PI * static_cast<double>(m) / static_cast<double>(n);
        cs[m] = std::cos(a);
        sn[m] = std::sin(a);
    }
    out_re.assign(n, 0.0);
    out_im.assign(n, 0.0);
    for (size_t k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        size_t idx = 0;
        for (size_t j = 0; j < n; ++j) {
            const double xr = in_re[j], xi = in_im[j];
            sr += xr * cs[idx] - xi * sn[idx];
            si += xr * sn[idx] + xi * cs[idx];
            idx += k;
            if (idx >= n)
                idx -= n;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
}

/// Relative L2 error ||got - ref|| / ||ref||.
double rel_l2_error(const float* got_re, const float* got_im, const std::vector<double>& ref_re,
                    const std::vector<double>& ref_im) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref_re.size(); ++i) {
        const double dr = got_re[i] - ref_re[i];
        const double di = got_im[i] - ref_im[i];
        num += dr * dr + di * di;
        den += ref_re[i] * ref_re[i] + ref_im[i] * ref_im[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

/// Error bound for a float FFT of length n: c * log2(n) * FLT_EPSILON.
double fft_error_bound(size_t n) {
    constexpr double c = 4.0;
    return c * std::max(1.0, std::log2(static_cast<double>(n))) * static_cast<double>(FLT_EPSILON);
}

void fill_random(std::vector<float>& re, std::vector<float>& im, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : re)
        v = dist(rng);
    for (auto& v : im)
        v = dist(rng);
}

} // namespace

// ---------- Reference-DFT accuracy (also pins the sign convention) ----------

class FFTReferenceTest : public ::testing::TestWithParam<size_t> {};

TEST_P(FFTReferenceTest, ForwardMatchesDoubleReference) {
    const size_t n = GetParam();
    std::vector<float> re(n), im(n);
    fill_random(re, im, 1234u + static_cast<uint32_t>(n));

    std::vector<double> ref_re, ref_im;
    reference_dft(re, im, ref_re, ref_im, -1);

    fft_forward(re.data(), im.data(), n);

    const double err = rel_l2_error(re.data(), im.data(), ref_re, ref_im);
    EXPECT_LE(err, fft_error_bound(n)) << "n=" << n << " rel L2 error " << err;
}

TEST_P(FFTReferenceTest, InverseMatchesDoubleReference) {
    const size_t n = GetParam();
    std::vector<float> re(n), im(n);
    fill_random(re, im, 99u + static_cast<uint32_t>(n));

    // Inverse = conjugate-direction DFT scaled by 1/n.
    std::vector<double> ref_re, ref_im;
    reference_dft(re, im, ref_re, ref_im, +1);
    for (size_t i = 0; i < n; ++i) {
        ref_re[i] /= static_cast<double>(n);
        ref_im[i] /= static_cast<double>(n);
    }

    fft_inverse(re.data(), im.data(), n);

    const double err = rel_l2_error(re.data(), im.data(), ref_re, ref_im);
    EXPECT_LE(err, fft_error_bound(n)) << "n=" << n << " rel L2 error " << err;
}

INSTANTIATE_TEST_SUITE_P(FFTRefSizes, FFTReferenceTest,
                         ::testing::Values(1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384));

// ---------- Sign convention ----------

TEST(FFTTest, ForwardOfPositiveExponentialLandsInBinK0) {
    // x[n] = exp(+2*pi*i*k0*n/N)  ->  X[k] = N * delta(k - k0) for the
    // exp(-2*pi*i*k*n/N) forward convention. A conjugated transform would put
    // the energy in bin N - k0 instead.
    for (const size_t n : {size_t{8}, size_t{64}, size_t{512}}) {
        const size_t k0 = 3;
        std::vector<float> re(n), im(n);
        for (size_t i = 0; i < n; ++i) {
            const double a = 2.0 * M_PI * static_cast<double>(k0 * i % n) / static_cast<double>(n);
            re[i] = static_cast<float>(std::cos(a));
            im[i] = static_cast<float>(std::sin(a));
        }

        fft_forward(re.data(), im.data(), n);

        EXPECT_NEAR(re[k0], static_cast<float>(n), 1e-3f * static_cast<float>(n)) << "n=" << n;
        EXPECT_NEAR(im[k0], 0.0f, 1e-3f * static_cast<float>(n)) << "n=" << n;
        for (size_t k = 0; k < n; ++k) {
            if (k == k0)
                continue;
            EXPECT_LE(std::hypot(re[k], im[k]), 1e-3f * static_cast<float>(n))
                << "leakage at bin " << k << " for n=" << n;
        }
    }
}

TEST(FFTTest, InverseOfKnownSpectrum) {
    // Spectrum with a single bin X[k0] = a + bi  ->  x[n] = (a + bi)/N * exp(+2*pi*i*k0*n/N)
    constexpr size_t n = 128;
    constexpr size_t k0 = 5;
    const double a = 2.0, b = -0.75;
    std::vector<float> re(n, 0.0f), im(n, 0.0f);
    re[k0] = static_cast<float>(a);
    im[k0] = static_cast<float>(b);

    fft_inverse(re.data(), im.data(), n);

    for (size_t i = 0; i < n; ++i) {
        const double ang = 2.0 * M_PI * static_cast<double>(k0 * i % n) / static_cast<double>(n);
        const double er = (a * std::cos(ang) - b * std::sin(ang)) / static_cast<double>(n);
        const double ei = (a * std::sin(ang) + b * std::cos(ang)) / static_cast<double>(n);
        EXPECT_NEAR(re[i], er, 1e-6) << "real at " << i;
        EXPECT_NEAR(im[i], ei, 1e-6) << "imag at " << i;
    }
}

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

    // Check reconstruction (relative L2 error against the original signal)
    std::vector<double> ref_re(original.begin(), original.end());
    std::vector<double> ref_im(n, 0.0);
    const double err = rel_l2_error(data.real, data.imag, ref_re, ref_im);
    EXPECT_LE(err, 2.0 * fft_error_bound(n)) << "n=" << n << " round-trip error " << err;
}

INSTANTIATE_TEST_SUITE_P(FFTSizes, FFTRoundTripTest,
                         ::testing::Values(1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384));

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
        EXPECT_NEAR(data.real[i], 1.0f, 1e-5f) << "Real part not 1.0 at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-5f) << "Imag part not 0.0 at bin " << i;
    }
}

// ---------- Known sinusoid test ----------

TEST(FFTTest, SingleSinusoidPeak) {
    constexpr size_t n = 256;
    constexpr size_t freq_bin = 10; // Frequency we'll inject

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
        float mag = std::sqrt(data.real[i] * data.real[i] + data.imag[i] * data.imag[i]);
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

    // DC bin should be n, all others 0 (tolerances scale with n)
    const float tol = static_cast<float>(n) * 4.0f * FLT_EPSILON;
    EXPECT_NEAR(data.real[0], static_cast<float>(n), tol);
    EXPECT_NEAR(data.imag[0], 0.0f, tol);

    for (size_t i = 1; i < n; ++i) {
        EXPECT_NEAR(data.real[i], 0.0f, tol) << "Nonzero real at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, tol) << "Nonzero imag at bin " << i;
    }
}

// ---------- Energy preservation (Parseval's theorem) ----------

TEST(FFTTest, EnergyPreservation) {
    constexpr size_t n = 512;
    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    // Random-ish signal
    for (size_t i = 0; i < n; ++i) {
        data.real[i] = static_cast<float>(std::sin(0.1 * static_cast<double>(i)) *
                                          std::cos(0.03 * static_cast<double>(i)));
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

    EXPECT_NEAR(freq_energy, time_energy, time_energy * 1e-4) << "Parseval's theorem violated";
}

// ---------- Size / argument validation ----------

TEST(FFTValidationTest, NonPowerOfTwoThrowsAndLeavesDataUntouched) {
    // Exact-size heap vectors: with the old assert-only check these sizes
    // indexed past the end (log2_of(12) = 4 -> bit_reverse up to 15).
    for (const size_t n : {size_t{3}, size_t{5}, size_t{6}, size_t{7}, size_t{12}, size_t{24},
                           size_t{100}, size_t{1000}, size_t{1023}, size_t{1025}}) {
        std::vector<float> re(n, 1.0f), im(n, 2.0f);
        EXPECT_THROW(fft_forward(re.data(), im.data(), n), std::invalid_argument) << "n=" << n;
        EXPECT_THROW(fft_inverse(re.data(), im.data(), n), std::invalid_argument) << "n=" << n;
        EXPECT_EQ(re, std::vector<float>(n, 1.0f));
        EXPECT_EQ(im, std::vector<float>(n, 2.0f));
    }
}

TEST(FFTValidationTest, HugeSizesThrowInsteadOfHanging) {
    // n >= 2^32 used to hit `1u << result` (UB, infinite loop). Not a power of
    // two or beyond the supported maximum, so both must throw immediately,
    // before any pointer is dereferenced.
    float dummy_re = 0.0f, dummy_im = 0.0f;
    const size_t big = size_t{1} << 40;
    EXPECT_THROW(fft_forward(&dummy_re, &dummy_im, big), std::invalid_argument);
    EXPECT_THROW(fft_inverse(&dummy_re, &dummy_im, big + 1), std::invalid_argument);
    EXPECT_THROW(fft_forward(&dummy_re, &dummy_im, std::numeric_limits<size_t>::max()),
                 std::invalid_argument);
}

TEST(FFTValidationTest, NullAndAliasedPointersThrow) {
    std::vector<float> buf(16, 0.0f);
    EXPECT_THROW(fft_forward(nullptr, buf.data(), 16), std::invalid_argument);
    EXPECT_THROW(fft_forward(buf.data(), nullptr, 16), std::invalid_argument);
    EXPECT_THROW(fft_inverse(buf.data(), buf.data(), 16), std::invalid_argument);
}

TEST(FFTValidationTest, EmptyAndSingleElementAreNoOps) {
    // n == 0: documented no-op, pointers are not inspected.
    EXPECT_NO_THROW(fft_forward(nullptr, nullptr, 0));
    EXPECT_NO_THROW(fft_inverse(nullptr, nullptr, 0));

    // n == 1: identity in both directions.
    float re = 3.5f, im = -1.25f;
    fft_forward(&re, &im, 1);
    EXPECT_EQ(re, 3.5f);
    EXPECT_EQ(im, -1.25f);
    fft_inverse(&re, &im, 1);
    EXPECT_EQ(re, 3.5f);
    EXPECT_EQ(im, -1.25f);
}

// ---------- Concurrent use of the thread-local twiddle cache ----------

TEST(FFTThreadingTest, ConcurrentTransformsAreCorrectAndDeterministic) {
    const std::vector<size_t> sizes = {2, 8, 16, 64, 256, 1024, 4096};

    // Single-threaded golden outputs.
    std::vector<std::vector<float>> in_re(sizes.size()), in_im(sizes.size());
    std::vector<std::vector<float>> gold_re(sizes.size()), gold_im(sizes.size());
    for (size_t s = 0; s < sizes.size(); ++s) {
        in_re[s].resize(sizes[s]);
        in_im[s].resize(sizes[s]);
        fill_random(in_re[s], in_im[s], 7u + static_cast<uint32_t>(s));
        gold_re[s] = in_re[s];
        gold_im[s] = in_im[s];
        fft_forward(gold_re[s].data(), gold_im[s].data(), sizes[s]);
    }

    constexpr int kThreads = 8;
    constexpr int kIterations = 20;
    std::vector<int> failures(kThreads, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int it = 0; it < kIterations; ++it) {
                for (size_t k = 0; k < sizes.size(); ++k) {
                    // Different threads walk the sizes in different orders so
                    // their thread-local caches fill in different sequences.
                    const size_t s = (k + static_cast<size_t>(t)) % sizes.size();
                    std::vector<float> re = in_re[s], im = in_im[s];
                    fft_forward(re.data(), im.data(), sizes[s]);
                    if (re != gold_re[s] || im != gold_im[s])
                        ++failures[static_cast<size_t>(t)];
                    fft_inverse(re.data(), im.data(), sizes[s]);
                    // forward+inverse reproduces the input to a few ulps relative to its norm
                    double num = 0.0, den = 0.0;
                    for (size_t i = 0; i < sizes[s]; ++i) {
                        const double dr = re[i] - in_re[s][i];
                        const double di = im[i] - in_im[s][i];
                        num += dr * dr + di * di;
                        den +=
                            double(in_re[s][i]) * in_re[s][i] + double(in_im[s][i]) * in_im[s][i];
                    }
                    if (std::sqrt(num / den) > 2.0 * fft_error_bound(sizes[s]))
                        ++failures[static_cast<size_t>(t)];
                }
            }
        });
    }
    for (auto& th : threads)
        th.join();
    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(failures[static_cast<size_t>(t)], 0) << "thread " << t;
    }
}
