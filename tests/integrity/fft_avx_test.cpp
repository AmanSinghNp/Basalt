#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/fft.hpp"
#include "basalt/kernel/cpu_state.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/transpose_avx.hpp"
#include "basalt/kernel/transpose.hpp"
#include "basalt/arena.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace basalt::kernel;

namespace {

/// Saves MXCSR on construction and restores it on destruction, so a test that
/// flips FTZ/DAZ cannot change the floating-point environment of later tests.
class MxcsrGuard {
public:
    MxcsrGuard() : m_saved(_mm_getcsr()) {}
    ~MxcsrGuard() { _mm_setcsr(m_saved); }
    MxcsrGuard(const MxcsrGuard&) = delete;
    MxcsrGuard& operator=(const MxcsrGuard&) = delete;

private:
    unsigned int m_saved;
};

/// Double-precision O(n^2) DFT; sign = -1 forward, +1 inverse direction (unscaled).
void reference_dft(const std::vector<float>& in_re, const std::vector<float>& in_im,
                   std::vector<double>& out_re, std::vector<double>& out_im, int sign) {
    const size_t n = in_re.size();
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
            if (idx >= n) idx -= n;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
}

double rel_l2_error(const float* got_re, const float* got_im,
                    const std::vector<double>& ref_re, const std::vector<double>& ref_im) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref_re.size(); ++i) {
        const double dr = got_re[i] - ref_re[i];
        const double di = got_im[i] - ref_im[i];
        num += dr * dr + di * di;
        den += ref_re[i] * ref_re[i] + ref_im[i] * ref_im[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double rel_l2_error(const float* got_re, const float* got_im,
                    const float* ref_re, const float* ref_im, size_t n) {
    std::vector<double> rr(ref_re, ref_re + n), ri(ref_im, ref_im + n);
    return rel_l2_error(got_re, got_im, rr, ri);
}

/// c * log2(n) * FLT_EPSILON error bound for a float FFT of length n.
double fft_error_bound(size_t n) {
    constexpr double c = 4.0;
    return c * std::max(1.0, std::log2(static_cast<double>(n))) * static_cast<double>(FLT_EPSILON);
}

void fill_random(std::vector<float>& re, std::vector<float>& im, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : re) v = dist(rng);
    for (auto& v : im) v = dist(rng);
}

} // namespace

// ---------- FTZ/DAZ test ----------

TEST(CPUStateTest, FlushToZeroWorks) {
    MxcsrGuard guard;  // restores the caller's MXCSR even if an assertion fails
    set_flush_to_zero();
    EXPECT_TRUE(check_ftz_daz());
}

TEST(CPUStateTest, MxcsrGuardRestoresState) {
    const unsigned int before = _mm_getcsr();
    {
        MxcsrGuard guard;
        set_flush_to_zero();
    }
    EXPECT_EQ(_mm_getcsr(), before);
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

    const double err = rel_l2_error(avx_data.real, avx_data.imag,
                                    scalar_data.real, scalar_data.imag, n);
    EXPECT_LE(err, fft_error_bound(n)) << "AVX vs scalar rel L2 error for n=" << n;
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

    const std::vector<float> zeros(n, 0.0f);
    const double err = rel_l2_error(data.real, data.imag, original.data(), zeros.data(), n);
    EXPECT_LE(err, 2.0 * fft_error_bound(n)) << "AVX round-trip rel L2 error for n=" << n;
}

INSTANTIATE_TEST_SUITE_P(
    AvxFFTSizes, FFTAvxCrossValidation,
    ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384)
);

// ---------- AVX vs double-precision reference (also pins sign convention) ----------

class FFTAvxReferenceTest : public ::testing::TestWithParam<size_t> {};

TEST_P(FFTAvxReferenceTest, ForwardMatchesDoubleReference) {
    const size_t n = GetParam();
    std::vector<float> re(n), im(n);
    fill_random(re, im, 4242u + static_cast<uint32_t>(n));

    std::vector<double> ref_re, ref_im;
    reference_dft(re, im, ref_re, ref_im, -1);

    fft_forward_avx(re.data(), im.data(), n);

    const double err = rel_l2_error(re.data(), im.data(), ref_re, ref_im);
    EXPECT_LE(err, fft_error_bound(n)) << "n=" << n << " rel L2 error " << err;
}

TEST_P(FFTAvxReferenceTest, InverseMatchesDoubleReference) {
    const size_t n = GetParam();
    std::vector<float> re(n), im(n);
    fill_random(re, im, 77u + static_cast<uint32_t>(n));

    std::vector<double> ref_re, ref_im;
    reference_dft(re, im, ref_re, ref_im, +1);
    for (size_t i = 0; i < n; ++i) {
        ref_re[i] /= static_cast<double>(n);
        ref_im[i] /= static_cast<double>(n);
    }

    fft_inverse_avx(re.data(), im.data(), n);

    const double err = rel_l2_error(re.data(), im.data(), ref_re, ref_im);
    EXPECT_LE(err, fft_error_bound(n)) << "n=" << n << " rel L2 error " << err;
}

INSTANTIATE_TEST_SUITE_P(
    AvxRefSizes, FFTAvxReferenceTest,
    ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384)
);

TEST(FFTAvxTest, ForwardOfPositiveExponentialLandsInBinK0) {
    // x[n] = exp(+2*pi*i*k0*n/N) -> energy in bin k0 (not N - k0).
    for (const size_t n : {size_t{8}, size_t{16}, size_t{64}, size_t{512}, size_t{2048}}) {
        const size_t k0 = 3;
        std::vector<float> re(n), im(n);
        for (size_t i = 0; i < n; ++i) {
            const double a = 2.0 * M_PI * static_cast<double>(k0 * i % n) / static_cast<double>(n);
            re[i] = static_cast<float>(std::cos(a));
            im[i] = static_cast<float>(std::sin(a));
        }

        fft_forward_avx(re.data(), im.data(), n);

        const float tol = 1e-3f * static_cast<float>(n);
        EXPECT_NEAR(re[k0], static_cast<float>(n), tol) << "n=" << n;
        EXPECT_NEAR(im[k0], 0.0f, tol) << "n=" << n;
        for (size_t k = 0; k < n; ++k) {
            if (k == k0) continue;
            EXPECT_LE(std::hypot(re[k], im[k]), tol) << "leakage at bin " << k << " for n=" << n;
        }
    }
}

TEST(FFTAvxTest, InverseOfKnownSpectrum) {
    constexpr size_t n = 256;
    constexpr size_t k0 = 37;
    std::vector<float> re(n, 0.0f), im(n, 0.0f);
    re[k0] = 1.5f;
    im[k0] = 0.5f;

    fft_inverse_avx(re.data(), im.data(), n);

    for (size_t i = 0; i < n; ++i) {
        const double ang = 2.0 * M_PI * static_cast<double>(k0 * i % n) / static_cast<double>(n);
        const double er = (1.5 * std::cos(ang) - 0.5 * std::sin(ang)) / static_cast<double>(n);
        const double ei = (1.5 * std::sin(ang) + 0.5 * std::cos(ang)) / static_cast<double>(n);
        EXPECT_NEAR(re[i], er, 1e-6) << "real at " << i;
        EXPECT_NEAR(im[i], ei, 1e-6) << "imag at " << i;
    }
}

// ---------- Validation ----------

TEST(FFTAvxValidationTest, NonPowerOfTwoThrowsAndLeavesDataUntouched) {
    for (const size_t n : {size_t{3}, size_t{6}, size_t{12}, size_t{24}, size_t{48}, size_t{100},
                           size_t{1000}, size_t{4095}, size_t{4097}}) {
        std::vector<float> re(n, 1.0f), im(n, 2.0f);
        EXPECT_THROW(fft_forward_avx(re.data(), im.data(), n), std::invalid_argument) << "n=" << n;
        EXPECT_THROW(fft_inverse_avx(re.data(), im.data(), n), std::invalid_argument) << "n=" << n;
        EXPECT_EQ(re, std::vector<float>(n, 1.0f));
        EXPECT_EQ(im, std::vector<float>(n, 2.0f));
    }
}

TEST(FFTAvxValidationTest, HugeNullAndAliasedArgumentsThrow) {
    float a = 0.0f, b = 0.0f;
    EXPECT_THROW(fft_forward_avx(&a, &b, size_t{1} << 40), std::invalid_argument);
    EXPECT_THROW(fft_inverse_avx(&a, &b, std::numeric_limits<size_t>::max()), std::invalid_argument);

    std::vector<float> buf(32, 0.0f);
    EXPECT_THROW(fft_forward_avx(nullptr, buf.data(), 32), std::invalid_argument);
    EXPECT_THROW(fft_inverse_avx(buf.data(), nullptr, 32), std::invalid_argument);
    EXPECT_THROW(fft_forward_avx(buf.data(), buf.data(), 32), std::invalid_argument);
}

TEST(FFTAvxValidationTest, EmptyAndSingleElementAreNoOps) {
    EXPECT_NO_THROW(fft_forward_avx(nullptr, nullptr, 0));
    EXPECT_NO_THROW(fft_inverse_avx(nullptr, nullptr, 0));
    float re = -2.0f, im = 0.5f;
    fft_forward_avx(&re, &im, 1);
    fft_inverse_avx(&re, &im, 1);
    EXPECT_EQ(re, -2.0f);
    EXPECT_EQ(im, 0.5f);
}

// ---------- AVX transpose validation ----------

TEST(TransposeAvxValidationTest, NullAndOverlappingBuffersThrow) {
    std::vector<float> a(16 * 16, 1.0f), b(16 * 16, 0.0f);
    EXPECT_THROW(transpose_tiled_avx(nullptr, b.data(), 16, 16), std::invalid_argument);
    EXPECT_THROW(transpose_tiled_avx(a.data(), nullptr, 16, 16), std::invalid_argument);
    EXPECT_THROW(transpose_tiled_avx(a.data(), a.data(), 16, 16), std::invalid_argument);
    // Partial overlap: dst starts inside src.
    std::vector<float> big(16 * 16 + 8, 0.0f);
    EXPECT_THROW(transpose_tiled_avx(big.data(), big.data() + 8, 16, 16), std::invalid_argument);
    EXPECT_THROW(transpose_tiled_avx(big.data() + 8, big.data(), 16, 16), std::invalid_argument);
    // Adjacent, non-overlapping ranges are fine.
    std::vector<float> two(2 * 16 * 16, 3.0f);
    EXPECT_NO_THROW(transpose_tiled_avx(two.data(), two.data() + 16 * 16, 16, 16));
}

TEST(TransposeAvxValidationTest, EmptyAndOverflowingShapes) {
    EXPECT_NO_THROW(transpose_tiled_avx(nullptr, nullptr, 0, 8));
    EXPECT_NO_THROW(transpose_tiled_avx(nullptr, nullptr, 8, 0));
    float x = 0.0f, y = 0.0f;
    EXPECT_THROW(transpose_tiled_avx(&x, &y, std::numeric_limits<size_t>::max() / 2, 4),
                 std::invalid_argument);
}

TEST(TransposeAvxValidationTest, ExtremeTileSizesStillCorrect) {
    constexpr size_t rows = 24, cols = 40;
    std::vector<float> src(rows * cols);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i);
    std::vector<float> expect(rows * cols);
    transpose_tiled(src.data(), expect.data(), rows, cols);
    for (const size_t tile : {size_t{0}, size_t{1}, std::numeric_limits<size_t>::max()}) {
        std::vector<float> dst(rows * cols);
        transpose_tiled_avx(src.data(), dst.data(), rows, cols, tile);
        EXPECT_EQ(dst, expect) << "tile " << tile;
    }
}

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

TEST(TransposeAvxTest, CacheTileSizeVariants) {
    constexpr size_t rows = 96;
    constexpr size_t cols = 160;
    std::vector<float> src(rows * cols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
    }

    std::vector<float> dst_scalar(rows * cols);
    transpose_tiled(src.data(), dst_scalar.data(), rows, cols);

    const std::vector<size_t> tile_sizes = {8, 17, 64, 128, 192};
    for (size_t tile : tile_sizes) {
        std::vector<float> dst_avx(rows * cols);
        transpose_tiled_avx(src.data(), dst_avx.data(), rows, cols, tile);
        EXPECT_EQ(dst_avx, dst_scalar) << "Mismatch with cache tile size " << tile;
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
