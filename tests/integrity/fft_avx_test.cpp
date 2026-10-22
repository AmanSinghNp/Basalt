#include "basalt/arena.hpp"
#include "basalt/internal/fft_plan.hpp"
#include "basalt/internal/simd_dispatch.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/cpu_features.hpp"
#include "basalt/kernel/cpu_state.hpp"
#include "basalt/kernel/fft.hpp"
#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/transpose.hpp"
#include "basalt/kernel/transpose_avx.hpp"
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(__linux__)
#include <unistd.h>
#endif
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
            if (idx >= n)
                idx -= n;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
}

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

double rel_l2_error(const float* got_re, const float* got_im, const float* ref_re,
                    const float* ref_im, size_t n) {
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
    for (auto& v : re)
        v = dist(rng);
    for (auto& v : im)
        v = dist(rng);
}

} // namespace

// ---------- FTZ/DAZ test ----------

TEST(CPUStateTest, FlushToZeroWorks) {
    MxcsrGuard guard; // restores the caller's MXCSR even if an assertion fails
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
        float val = static_cast<float>(std::sin(2.0 * M_PI * 5.0 * t) +
                                       0.3 * std::cos(2.0 * M_PI * 13.0 * t));
        scalar_data.real[i] = val;
        scalar_data.imag[i] = 0.0f;
        avx_data.real[i] = val;
        avx_data.imag[i] = 0.0f;
    }

    fft_forward(scalar_data.real, scalar_data.imag, n);
    fft_forward_avx(avx_data.real, avx_data.imag, n);

    const double err =
        rel_l2_error(avx_data.real, avx_data.imag, scalar_data.real, scalar_data.imag, n);
    EXPECT_LE(err, fft_error_bound(n)) << "AVX vs scalar rel L2 error for n=" << n;
}

TEST_P(FFTAvxCrossValidation, RoundTripPreservesData) {
    size_t n = GetParam();

    basalt::MemoryArena arena(n * 2 * sizeof(float) + 4096);
    ComplexSoA data(arena, n);

    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n);
        data.real[i] = static_cast<float>(std::sin(2.0 * M_PI * 3.0 * t) +
                                          0.5 * std::cos(2.0 * M_PI * 7.0 * t));
        data.imag[i] = 0.0f;
    }

    std::vector<float> original(data.real, data.real + n);

    fft_forward_avx(data.real, data.imag, n);
    fft_inverse_avx(data.real, data.imag, n);

    const std::vector<float> zeros(n, 0.0f);
    const double err = rel_l2_error(data.real, data.imag, original.data(), zeros.data(), n);
    EXPECT_LE(err, 2.0 * fft_error_bound(n)) << "AVX round-trip rel L2 error for n=" << n;
}

INSTANTIATE_TEST_SUITE_P(AvxFFTSizes, FFTAvxCrossValidation,
                         ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384));

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

INSTANTIATE_TEST_SUITE_P(AvxRefSizes, FFTAvxReferenceTest,
                         ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384));

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
            if (k == k0)
                continue;
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
    EXPECT_THROW(fft_inverse_avx(&a, &b, std::numeric_limits<size_t>::max()),
                 std::invalid_argument);

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
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<float>(i);
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
        EXPECT_FLOAT_EQ(dst_avx[i], dst_scalar[i]) << "Mismatch at flat index " << i;
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
        data.real[i] = static_cast<float>(std::sin(0.1 * static_cast<double>(i)) *
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

// ---------- Runtime CPU features and dispatch ----------

namespace {

std::string lower_env(const char* name) {
    const char* v = std::getenv(name);
    std::string out = v ? v : "";
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

const char* cap_name(SimdMode m) {
    switch (m) {
    case SimdMode::Auto:
        return "auto";
    case SimdMode::AVX2:
        return "avx2";
    case SimdMode::AVX512:
        return "avx512";
    case SimdMode::Scalar:
        return "scalar";
    }
    return "?";
}

} // namespace

TEST(CpuFeaturesTest, IsStableAndThreadSafe) {
    const CpuFeatures* first = &cpu_features();
    EXPECT_EQ(first, &cpu_features());
    std::vector<const CpuFeatures*> seen(8, nullptr);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < seen.size(); ++t) {
        threads.emplace_back([&seen, t] { seen[t] = &cpu_features(); });
    }
    for (auto& th : threads)
        th.join();
    for (const CpuFeatures* p : seen) {
        EXPECT_EQ(p, first);
    }
}

TEST(CpuFeaturesTest, EnvOverrideRespected) {
    // Also run as a child process with BASALT_SIMD_MAX set (see the test below).
    const CpuFeatures& f = cpu_features();
    const std::string cap = lower_env("BASALT_SIMD_MAX");
    if (cap == "scalar") {
        EXPECT_FALSE(f.avx2);
        EXPECT_FALSE(f.fma);
        EXPECT_FALSE(f.avx512f);
        EXPECT_FALSE(f.avx512dq);
        EXPECT_FALSE(f.avx512bw);
        EXPECT_FALSE(f.avx512vl);
        EXPECT_FALSE(f.can_use_avx2());
    } else if (cap == "avx2") {
        EXPECT_FALSE(f.avx512f);
        EXPECT_FALSE(f.avx512dq);
        EXPECT_FALSE(f.avx512bw);
        EXPECT_FALSE(f.avx512vl);
        EXPECT_FALSE(f.can_use_avx512());
    }
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    // Whatever is reported must really be supported by the CPU: the cap only removes features.
    if (f.avx2) {
        EXPECT_TRUE(__builtin_cpu_supports("avx2"));
    }
    if (f.fma) {
        EXPECT_TRUE(__builtin_cpu_supports("fma"));
    }
    if (f.avx512f) {
        EXPECT_TRUE(__builtin_cpu_supports("avx512f"));
    }
    if (f.avx512dq) {
        EXPECT_TRUE(__builtin_cpu_supports("avx512dq"));
    }
    if (f.avx512bw) {
        EXPECT_TRUE(__builtin_cpu_supports("avx512bw"));
    }
    if (f.avx512vl) {
        EXPECT_TRUE(__builtin_cpu_supports("avx512vl"));
    }
    if (cap != "scalar" && cap != "avx2") {
        // No effective cap: detection equals the compiler's own answer.
        EXPECT_EQ(f.avx2, static_cast<bool>(__builtin_cpu_supports("avx2")));
        EXPECT_EQ(f.fma, static_cast<bool>(__builtin_cpu_supports("fma")));
        EXPECT_EQ(f.avx512f, static_cast<bool>(__builtin_cpu_supports("avx512f")));
        EXPECT_EQ(f.avx512dq, static_cast<bool>(__builtin_cpu_supports("avx512dq")));
        EXPECT_EQ(f.avx512bw, static_cast<bool>(__builtin_cpu_supports("avx512bw")));
        EXPECT_EQ(f.avx512vl, static_cast<bool>(__builtin_cpu_supports("avx512vl")));
    }
#endif
    // The derived predicates agree with the raw flags.
    EXPECT_EQ(f.can_use_avx2(), f.avx2 && f.fma);
    EXPECT_EQ(f.can_use_avx512(), f.avx2 && f.fma && f.avx512f && f.avx512dq);
}

#if defined(__linux__)
TEST(CpuFeaturesTest, EnvVariableIsParsedByChildProcess) {
    // cpu_features() is computed once per process, so the parsing of
    // BASALT_SIMD_MAX is exercised in fresh child processes of this binary.
    char exe[4096] = {};
    const ssize_t len = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    ASSERT_GT(len, 0);
    for (const char* value : {"scalar", "SCALAR", "avx2", "Avx2", "avx512", "bogus", ""}) {
        const std::string cmd =
            std::string("BASALT_SIMD_MAX='") + value + "' '" + exe +
            "' --gtest_filter=CpuFeaturesTest.EnvOverrideRespected >/dev/null 2>&1";
        EXPECT_EQ(std::system(cmd.c_str()), 0) << "BASALT_SIMD_MAX=" << value;
    }
}
#endif

TEST(CpuFeaturesTest, DispatchLevelsHonourCaps) {
    using namespace basalt::kernel::detail;
    const SimdLevel hardware = max_simd_level();
    {
        ScopedSimdCap cap(SimdMode::Scalar);
        EXPECT_EQ(max_simd_level(), SimdLevel::Scalar);
        EXPECT_FALSE(use_avx2());
        for (SimdMode m : {SimdMode::Auto, SimdMode::AVX2, SimdMode::AVX512, SimdMode::Scalar}) {
            EXPECT_EQ(resolve_simd_level(m), SimdLevel::Scalar) << cap_name(m);
        }
    }
    {
        ScopedSimdCap cap(SimdMode::AVX2);
        EXPECT_LE(max_simd_level(), SimdLevel::AVX2);
        EXPECT_EQ(resolve_simd_level(SimdMode::AVX512), resolve_simd_level(SimdMode::AVX2));
        EXPECT_EQ(resolve_simd_level(SimdMode::Scalar), SimdLevel::Scalar);
    }
    EXPECT_EQ(max_simd_level(), hardware); // scoped caps restore the previous state
    EXPECT_EQ(simd_cap_for_testing(), SimdMode::Auto);
    // Explicit requests never exceed the hardware level; Scalar is always honoured.
    EXPECT_LE(resolve_simd_level(SimdMode::AVX2), SimdLevel::AVX2);
    EXPECT_EQ(resolve_simd_level(SimdMode::Scalar), SimdLevel::Scalar);
    EXPECT_EQ(resolve_simd_level(SimdMode::Auto), hardware);
    // Nested caps restore in order.
    {
        ScopedSimdCap outer(SimdMode::Scalar);
        {
            ScopedSimdCap inner(SimdMode::AVX2);
            EXPECT_EQ(simd_cap_for_testing(), SimdMode::AVX2);
        }
        EXPECT_EQ(simd_cap_for_testing(), SimdMode::Scalar);
    }
}

// ---------- Every kernel under every cap ----------

class SimdCapParamTest : public ::testing::TestWithParam<SimdMode> {};

INSTANTIATE_TEST_SUITE_P(Caps, SimdCapParamTest,
                         ::testing::Values(SimdMode::Scalar, SimdMode::AVX2, SimdMode::AVX512),
                         [](const ::testing::TestParamInfo<SimdMode>& param_info) {
                             return std::string(cap_name(param_info.param));
                         });

TEST_P(SimdCapParamTest, ExplicitFftMatchesReferenceAndRoundTrips) {
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    for (size_t n = 1; n <= 1024; n *= 2) {
        std::vector<float> in_re(n), in_im(n);
        fill_random(in_re, in_im, static_cast<uint32_t>(n * 7 + 1));

        std::vector<double> ref_re, ref_im;
        reference_dft(in_re, in_im, ref_re, ref_im, -1);

        std::vector<float> re = in_re, im = in_im;
        fft_forward_avx(re.data(), im.data(), n);
        EXPECT_LE(rel_l2_error(re.data(), im.data(), ref_re, ref_im), fft_error_bound(n))
            << "forward n=" << n << " cap=" << cap_name(GetParam());

        fft_inverse_avx(re.data(), im.data(), n);
        EXPECT_LE(rel_l2_error(re.data(), im.data(), in_re.data(), in_im.data(), n),
                  2.0 * fft_error_bound(n))
            << "round trip n=" << n << " cap=" << cap_name(GetParam());
    }
}

TEST_P(SimdCapParamTest, LargeFftAgreesWithScalarKernel) {
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    for (const size_t n : {size_t{4096}, size_t{65536}}) {
        std::vector<float> in_re(n), in_im(n);
        fill_random(in_re, in_im, 99);
        std::vector<float> ref_re = in_re, ref_im = in_im;
        fft_forward(ref_re.data(), ref_im.data(), n);

        std::vector<float> re = in_re, im = in_im;
        fft_forward_avx(re.data(), im.data(), n);
        EXPECT_LE(rel_l2_error(re.data(), im.data(), ref_re.data(), ref_im.data(), n),
                  2.0 * fft_error_bound(n))
            << "n=" << n;

        if (GetParam() == SimdMode::Scalar) {
            // The capped "AVX" entry point runs the very same scalar kernel.
            EXPECT_EQ(re, ref_re) << "n=" << n;
            EXPECT_EQ(im, ref_im) << "n=" << n;
        }
    }
}

TEST_P(SimdCapParamTest, ColumnStripsMatchPerColumnFft) {
    // fft_cols_strips (the transpose-free column pass of fft2d) against the 1D
    // scalar kernel applied to each extracted column, for every strip width the
    // AVX2 path dispatches on, both directions and a non-trivial scale.
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    struct Case {
        size_t rows, cols, width;
    };
    const Case cases[] = {
        {2, 4, 4},      {8, 2, 2},      {16, 1, 1},     {4, 8, 8},     {64, 8, 8},
        {64, 16, 16},   {128, 32, 32},  {256, 64, 64},  {32, 128, 64}, {512, 64, 32},
        {1024, 64, 16}, {4096, 64, 64}, {2048, 128, 8}, {1, 16, 16},
    };
    for (const Case& c : cases) {
        for (const bool inverse : {false, true}) {
            for (const float scale : {1.0f, 0.25f}) {
                const size_t total = c.rows * c.cols;
                std::vector<float> in_re(total), in_im(total);
                fill_random(in_re, in_im, static_cast<uint32_t>(c.rows * 13 + c.cols + c.width));

                // Reference: per-column scalar FFT (inverse = conj(fft(conj(x)))), then scale.
                std::vector<float> ref_re = in_re, ref_im = in_im;
                for (size_t col = 0; col < c.cols; ++col) {
                    std::vector<float> cr(c.rows), ci(c.rows);
                    for (size_t r = 0; r < c.rows; ++r) {
                        cr[r] = in_re[r * c.cols + col];
                        ci[r] = inverse ? -in_im[r * c.cols + col] : in_im[r * c.cols + col];
                    }
                    fft_forward(cr.data(), ci.data(), c.rows);
                    for (size_t r = 0; r < c.rows; ++r) {
                        ref_re[r * c.cols + col] = cr[r] * scale;
                        ref_im[r * c.cols + col] = (inverse ? -ci[r] : ci[r]) * scale;
                    }
                }

                std::vector<float> re = in_re, im = in_im;
                std::vector<float> buf_re(c.rows * c.width, 7.0f), buf_im(c.rows * c.width, -7.0f);
                basalt::internal::fft_cols_strips(re.data(), im.data(), c.rows, c.cols, inverse,
                                                  scale, buf_re.data(), buf_im.data(), c.width);
                EXPECT_LE(rel_l2_error(re.data(), im.data(), ref_re.data(), ref_im.data(), total),
                          2.0 * fft_error_bound(c.rows))
                    << c.rows << "x" << c.cols << " width " << c.width
                    << (inverse ? " inverse" : " forward") << " scale " << scale
                    << " cap=" << cap_name(GetParam());
            }
        }
    }

    // The advertised strip width is a power of two that fits the matrix.
    for (const size_t cols : {size_t{1}, size_t{2}, size_t{8}, size_t{64}, size_t{4096}}) {
        const size_t w = basalt::internal::fft_cols_strip_width(256, cols);
        EXPECT_GE(w, 1u);
        EXPECT_LE(w, cols);
        EXPECT_EQ(w & (w - 1), 0u);
    }
}

TEST_P(SimdCapParamTest, ExplicitTransposeMatchesScalar) {
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    const std::pair<size_t, size_t> shapes[] = {{1, 1},    {1, 9},    {9, 1},   {7, 13},
                                                {8, 8},    {16, 24},  {33, 65}, {100, 37},
                                                {128, 64}, {257, 129}};
    for (const auto& shape : shapes) {
        const size_t rows = shape.first, cols = shape.second;
        std::vector<float> src(rows * cols);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = static_cast<float>(i) * 0.25f;
        std::vector<float> expect(rows * cols);
        transpose_tiled(src.data(), expect.data(), rows, cols);
        for (const size_t tile : {size_t{0}, size_t{8}, size_t{17}, size_t{128}}) {
            std::vector<float> dst(rows * cols, -1.0f);
            transpose_tiled_avx(src.data(), dst.data(), rows, cols, tile);
            EXPECT_EQ(dst, expect)
                << rows << "x" << cols << " tile " << tile << " cap=" << cap_name(GetParam());
        }
    }
}

TEST_P(SimdCapParamTest, Explicit8x8BlockTranspose) {
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    constexpr size_t src_stride = 13, dst_stride = 11;
    std::vector<float> src(8 * src_stride);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<float>(i);
    std::vector<float> dst(8 * dst_stride, -1.0f);
    transpose_8x8_avx(src.data(), dst.data(), src_stride, dst_stride);
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            ASSERT_EQ(dst[j * dst_stride + i], src[i * src_stride + j]) << i << "," << j;
        }
        // Columns beyond the 8x8 block are not written.
        for (size_t j = 8; j < dst_stride; ++j) {
            ASSERT_EQ(dst[i * dst_stride + j], -1.0f);
        }
    }
}

TEST_P(SimdCapParamTest, ExplicitCallsStillValidateArguments) {
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    std::vector<float> buf(64, 0.0f);
    EXPECT_THROW(fft_forward_avx(buf.data(), buf.data() + 1, 3), std::invalid_argument);
    EXPECT_THROW(fft_inverse_avx(nullptr, buf.data(), 8), std::invalid_argument);
    EXPECT_THROW(transpose_tiled_avx(buf.data(), buf.data(), 8, 8), std::invalid_argument);
    EXPECT_NO_THROW(transpose_tiled_avx(nullptr, nullptr, 0, 8));
}
