#include <gtest/gtest.h>
#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/simd_mode.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <complex>
#include <numeric>
#include <random>

using namespace basalt::filter;
using namespace basalt::kernel;

namespace {

// Helper to fill data with a specific velocity event
void fill_event(ComplexSoA& data, size_t rows, size_t cols, double dt, double dx, double velocity, float amplitude) {
    // In f-k domain, a linear event with velocity V maps to a line f = V*k

    double df = 1.0 / (static_cast<double>(rows) * dt);
    double dk = 1.0 / (static_cast<double>(cols) * dx);

    for (size_t i = 0; i < rows; ++i) {
        double f = static_cast<double>(i) * df;
        // Handle Nyquist f
        if (i > rows/2) f = static_cast<double>((static_cast<int>(i) - static_cast<int>(rows))) * df;

        for (size_t j = 0; j < cols; ++j) {
            double k_val;
            if (j <= cols/2) k_val = static_cast<double>(j) * dk;
            else k_val = static_cast<double>((static_cast<int>(j) - static_cast<int>(cols))) * dk;

            // Avoid singularity
            if (std::abs(k_val) < 1e-6) continue;

            double v_point = f / k_val;

            // If v_point is within 10% of target velocity
            if (std::abs(v_point - velocity) < std::abs(velocity) * 0.1) {
                data.real[i * cols + j] += amplitude;
                // data.imag remains 0
            }
        }
    }
}

// Signed frequency / wavenumber of bin `index` on an axis of `size` bins with spacing `step`.
// Same wrapping convention as the filter (indices above size/2 are negative).
double wrapped_axis(size_t index, size_t size, double step) {
    const long long signed_index = index <= size / 2
        ? static_cast<long long>(index)
        : static_cast<long long>(index) - static_cast<long long>(size);
    return static_cast<double>(signed_index) * step;
}

// Scalar double-precision reference for the mute weight of bin (f, k).
// v = f / k is the apparent velocity; k == 0 bins are always kept.
double reference_weight(double f, double k, double vmin, double vmax, double taper_width) {
    if (k == 0.0) {
        return 1.0;
    }
    const double v = f / k;
    const double s = 10.0 / taper_width;
    const double rise = 1.0 / (1.0 + std::exp(-(v - vmin) * s));
    const double fall = 1.0 / (1.0 + std::exp(-(vmax - v) * s));
    return 1.0 - rise * fall;
}

// Expected weight of grid bin (i, j) including the Nyquist convention: the Nyquist row and
// column (even sizes) are their own alias of both signs, so their weight is the mean of the
// weights at f and -f (equivalently v and -v). All other bins use reference_weight directly.
double expected_weight(size_t i, size_t j, size_t rows, size_t cols,
                       double df, double dk, const FKParams& params) {
    const double f = wrapped_axis(i, rows, df);
    const double k = wrapped_axis(j, cols, dk);
    const bool nyquist_row = rows % 2 == 0 && i == rows / 2;
    const bool nyquist_col = cols % 2 == 0 && j == cols / 2;
    const double direct = reference_weight(f, k, params.mute_vel_min, params.mute_vel_max,
                                           params.taper_width);
    if (!nyquist_row && !nyquist_col) {
        return direct;
    }
    const double mirror = reference_weight(-f, k, params.mute_vel_min, params.mute_vel_max,
                                           params.taper_width);
    return 0.5 * (direct + mirror);
}

// Reference for a symmetric band: weight = sigmoid((|v| - cut) * s) with cut = vmax.
// Written independently of the filter and without any epsilon.
double abs_velocity_weight(double v, double cut, double taper_width) {
    const double s = 10.0 / taper_width;
    return 1.0 / (1.0 + std::exp(-(std::abs(v) - cut) * s));
}

FKParams make_params(double vmin, double vmax, double taper_width) {
    FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = vmin;
    params.mute_vel_max = vmax;
    params.taper_width = taper_width;
    return params;
}

// Runs the filter on a spectrum of ones (real = 1, imag = 0), so the real output is the weight.
std::vector<float> filter_ones(size_t rows, size_t cols, const FKParams& params, SimdMode mode) {
    basalt::MemoryArena arena(rows * cols * sizeof(float) * 2 + 4096);
    ComplexSoA data(arena, rows * cols);
    std::fill_n(data.real, rows * cols, 1.0f);
    std::fill_n(data.imag, rows * cols, 0.0f);
    apply_fk_filter(data, rows, cols, params, mode);
    return std::vector<float>(data.real, data.real + rows * cols);
}

bool host_supports_avx512_filter() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

} // namespace

TEST(FKFilterTest, MutesLowVelocity) {
    basalt::MemoryArena arena(1024 * 1024 * 4);
    size_t rows = 256;
    size_t cols = 256;
    ComplexSoA data(arena, rows * cols);

    // Clear data
    std::fill_n(data.real, rows * cols, 0.0f);
    std::fill_n(data.imag, rows * cols, 0.0f);

    FKParams params;
    params.dt = 0.004;
    params.dx = 25.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0; // Mute everything inside [-1500, 1500]
    params.taper_width = 100.0;

    // Add "Ground Roll" event at 500 m/s (Should be muted)
    double noise_vel = 500.0;
    fill_event(data, rows, cols, params.dt, params.dx, noise_vel, 1.0f);

    // Add "Signal" event at 2500 m/s (Should pass)
    double signal_vel = 2500.0;
    fill_event(data, rows, cols, params.dt, params.dx, signal_vel, 1.0f);

    apply_fk_filter(data, rows, cols, params);

    // Check specific points
    double df = 1.0 / (static_cast<double>(rows) * params.dt);
    double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    for (size_t i = 1; i < rows/2; ++i) {
        for (size_t j = 1; j < cols/2; ++j) {
            double f = static_cast<double>(i) * df;
            double k_val = static_cast<double>(j) * dk;
            double v = f / k_val;

            size_t idx = i * cols + j;
            float mag = std::sqrt(data.real[idx]*data.real[idx] + data.imag[idx]*data.imag[idx]);

            if (std::abs(v - noise_vel) < noise_vel * 0.05) {
                // Should be muted (near 0)
                EXPECT_NEAR(mag, 0.0f, 1e-2) << "Noise at v=" << v << " not muted";
            }
            if (std::abs(v - signal_vel) < signal_vel * 0.05) {
                // Should pass (amplitude ~1.0)
                // Note: Taper might reduce it slightly if close to cut, but 2500 vs 1500 is far.
                EXPECT_NEAR(mag, 1.0f, 1e-2) << "Signal at v=" << v << " attenuated";
            }
        }
    }
}

TEST(FKFilterTest, PreservesDC) {
    // Every SIMD path has its own DC lane mask, so each one is checked explicitly.
    // Auto selects the best path on this host; the explicit modes fall back when unsupported.
    const SimdMode modes[] = {SimdMode::Auto, SimdMode::AVX2, SimdMode::AVX512};
    for (const SimdMode mode : modes) {
        basalt::MemoryArena arena(4096);
        size_t rows = 16;
        size_t cols = 16;
        ComplexSoA data(arena, rows * cols);
        std::fill_n(data.real, rows * cols, 0.0f);

        // Set DC
        data.real[0] = 100.0f;

        FKParams params;
        apply_fk_filter(data, rows, cols, params, mode);

        // DC (f = 0, k = 0) has no finite velocity and is always kept, so the data mean survives.
        EXPECT_EQ(data.real[0], 100.0f) << "simd mode " << static_cast<int>(mode);
    }
}

TEST(FKFilterTest, AsymmetricBandMutesOnlyIntendedSign) {
    constexpr size_t rows = 64;
    constexpr size_t cols = 64;
    const FKParams params = make_params(500.0, 2000.0, 50.0);
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    const std::vector<float> out = filter_ones(rows, cols, params, SimdMode::AVX2);

    double max_err = 0.0;
    size_t inside_positive = 0;
    size_t inside_negative = 0;
    for (size_t i = 0; i < rows; ++i) {
        const double f = wrapped_axis(i, rows, df);
        for (size_t j = 0; j < cols; ++j) {
            const double k = wrapped_axis(j, cols, dk);
            const double expected = expected_weight(i, j, rows, cols, df, dk, params);
            max_err = std::max(max_err, std::abs(static_cast<double>(out[i * cols + j]) - expected));

            // Nyquist row/column bins have no sign of their own (see
            // AsymmetricBandWeightsAreHermitianIncludingNyquist), so the one-sided checks below
            // apply to all other bins only.
            const bool nyquist = (rows % 2 == 0 && i == rows / 2) || (cols % 2 == 0 && j == cols / 2);
            if (k != 0.0 && !nyquist) {
                const double v = f / k;
                if (v > 600.0 && v < 1900.0) {
                    ++inside_positive;
                    EXPECT_LT(out[i * cols + j], 1e-3f) << "positive v=" << v << " not muted";
                }
                if (v < -600.0 && v > -1900.0) {
                    ++inside_negative;
                    EXPECT_GT(out[i * cols + j], 1.0f - 1e-3f) << "negative v=" << v << " muted";
                }
            }
        }
    }
    EXPECT_GT(inside_positive, 0u);
    EXPECT_GT(inside_negative, 0u);
    EXPECT_LT(max_err, 1e-4) << "Weights disagree with the scalar reference";

    // Explicit bins: v = +1000 is (f, k) = (4 * df, 5 * dk); flipping k gives v = -1000.
    const size_t pos_bin = 4 * cols + 5;
    const size_t neg_bin = 4 * cols + (cols - 5);
    EXPECT_NEAR(out[pos_bin], 0.0f, 1e-3f);
    EXPECT_NEAR(out[neg_bin], 1.0f, 1e-3f);
}

TEST(FKFilterTest, DCBinIsPreservedEvenInsideBand) {
    constexpr size_t rows = 32;
    constexpr size_t cols = 32;

    // Default symmetric band: DC (v = 0) lies inside it, yet must be kept.
    const std::vector<float> symmetric = filter_ones(rows, cols, FKParams{}, SimdMode::AVX2);
    EXPECT_EQ(symmetric[0], 1.0f);
    // The neighbouring bin (f = 0, k != 0) has v = 0 inside the band and is muted.
    EXPECT_LT(symmetric[1], 1e-3f);

    // Asymmetric band that contains v = 0 as well.
    const FKParams asymmetric = make_params(-100.0, 2000.0, 50.0);
    const std::vector<float> out = filter_ones(rows, cols, asymmetric, SimdMode::AVX2);
    EXPECT_EQ(out[0], 1.0f);
}

TEST(FKFilterTest, ZeroWavenumberColumnIsPreserved) {
    constexpr size_t rows = 64;
    constexpr size_t cols = 64;
    // Band wide enough to cover every finite velocity on the grid (|v| <= 40000 m/s).
    const FKParams wide = make_params(-1e5, 1e5, 100.0);
    const std::vector<float> out_wide = filter_ones(rows, cols, wide, SimdMode::AVX2);

    // Every finite-velocity bin is muted, while the whole k == 0 column is kept exactly.
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_EQ(out_wide[i * cols + 0], 1.0f) << "k == 0 bin at row " << i << " muted";
        for (size_t j = 1; j < cols; ++j) {
            EXPECT_LT(out_wide[i * cols + j], 1e-3f) << "bin (" << i << ", " << j << ") not muted";
        }
    }

    // Asymmetric band as well: the k == 0 column does not depend on the band.
    const FKParams asymmetric = make_params(500.0, 2000.0, 50.0);
    const std::vector<float> out = filter_ones(rows, cols, asymmetric, SimdMode::AVX2);
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_EQ(out[i * cols + 0], 1.0f) << "k == 0 bin at row " << i << " muted";
    }
}

TEST(FKFilterTest, EmptyBandIsNoOp) {
    constexpr size_t rows = 16;
    constexpr size_t cols = 24;
    constexpr size_t total = rows * cols;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> real(total);
    std::vector<float> imag(total);
    for (size_t idx = 0; idx < total; ++idx) {
        real[idx] = dist(rng);
        imag[idx] = dist(rng);
    }

    const FKParams cases[] = {
        make_params(500.0, 500.0, 50.0),     // min == max
        make_params(2000.0, 500.0, 50.0),    // min > max
        make_params(0.0, 0.0, 100.0),        // degenerate zero-width band at the origin
        make_params(-1500.0, 1500.0, 0.0),   // non-positive taper
    };

    for (const FKParams& params : cases) {
        basalt::MemoryArena arena(total * sizeof(float) * 2 + 4096);
        ComplexSoA data(arena, total);
        std::memcpy(data.real, real.data(), total * sizeof(float));
        std::memcpy(data.imag, imag.data(), total * sizeof(float));

        apply_fk_filter(data, rows, cols, params, SimdMode::AVX2);

        EXPECT_EQ(std::memcmp(data.real, real.data(), total * sizeof(float)), 0)
            << "real changed for min=" << params.mute_vel_min << " max=" << params.mute_vel_max
            << " taper=" << params.taper_width;
        EXPECT_EQ(std::memcmp(data.imag, imag.data(), total * sizeof(float)), 0)
            << "imag changed for min=" << params.mute_vel_min << " max=" << params.mute_vel_max
            << " taper=" << params.taper_width;
    }
}

TEST(FKFilterTest, SymmetricDefaultMatchesAbsVelocityReference) {
    constexpr size_t rows = 64;
    constexpr size_t cols = 64;
    const FKParams params = make_params(-1500.0, 1500.0, 100.0);
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    const std::vector<float> out = filter_ones(rows, cols, params, SimdMode::AVX2);

    double max_err = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        const double f = wrapped_axis(i, rows, df);
        for (size_t j = 0; j < cols; ++j) {
            const double k = wrapped_axis(j, cols, dk);
            if (k == 0.0) {
                // The k == 0 column has its own contract (weight exactly 1), covered by
                // ZeroWavenumberColumnIsPreserved, so it is not compared against |v| here.
                continue;
            }
            const double reference = abs_velocity_weight(f / k, params.mute_vel_max, params.taper_width);
            max_err = std::max(max_err, std::abs(static_cast<double>(out[i * cols + j]) - reference));
        }
    }
    // Checks the symmetric band against the |v| sigmoid in double precision, so the bound
    // covers only the fast-sigmoid approximation. It is not a comparison with an earlier binary.
    EXPECT_LT(max_err, 1e-4) << "Symmetric default drifted from the |v| sigmoid reference";

    // Far below the cut: muted. Far above: passed.
    for (size_t i = 0; i < rows; ++i) {
        const double f = wrapped_axis(i, rows, df);
        for (size_t j = 1; j < cols; ++j) {
            const double k = wrapped_axis(j, cols, dk);
            const double v = std::abs(f / k);
            if (v <= 300.0) {
                EXPECT_LT(out[i * cols + j], 1e-3f) << "|v|=" << v << " should be muted";
            } else if (v >= 2500.0) {
                EXPECT_GT(out[i * cols + j], 0.999f) << "|v|=" << v << " should pass";
            }
        }
    }
}

TEST(FKFilterTest, AsymmetricBandAvx2MatchesScalarReferenceOnRandomSpectrum) {
    constexpr size_t rows = 37;   // not a multiple of the vector width
    constexpr size_t cols = 50;   // tail lanes in both SIMD paths
    constexpr size_t total = rows * cols;
    const FKParams params = make_params(-300.0, 1200.0, 80.0);
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    basalt::MemoryArena arena(total * sizeof(float) * 2 + 4096);
    ComplexSoA data(arena, total);
    std::vector<float> real_in(total);
    std::vector<float> imag_in(total);
    for (size_t idx = 0; idx < total; ++idx) {
        real_in[idx] = dist(rng);
        imag_in[idx] = dist(rng);
    }
    std::memcpy(data.real, real_in.data(), total * sizeof(float));
    std::memcpy(data.imag, imag_in.data(), total * sizeof(float));

    apply_fk_filter(data, rows, cols, params, SimdMode::AVX2);

    double max_err = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const double w = expected_weight(i, j, rows, cols, df, dk, params);
            const size_t idx = i * cols + j;
            max_err = std::max(max_err, std::abs(static_cast<double>(data.real[idx]) - w * real_in[idx]));
            max_err = std::max(max_err, std::abs(static_cast<double>(data.imag[idx]) - w * imag_in[idx]));
        }
    }
    EXPECT_LT(max_err, 1e-4);
}

TEST(FKFilterSimdAgreementTest, Avx512WeightsMatchAvx2WithinTolerance) {
    if (!host_supports_avx512_filter()) {
        GTEST_SKIP() << "Host does not support AVX-512 filter path.";
    }

    struct Case {
        size_t rows;
        size_t cols;
        FKParams params;
    };
    const Case cases[] = {
        {37, 50, make_params(500.0, 2000.0, 50.0)},     // asymmetric band, tails in both paths
        {64, 64, make_params(-1500.0, 1500.0, 100.0)},  // symmetric default
        {64, 64, make_params(-300.0, 1200.0, 80.0)},    // asymmetric band containing v = 0
        {9, 21, make_params(-1e5, 1e5, 100.0)},         // band covering every finite v
    };

    for (const Case& c : cases) {
        const size_t total = c.rows * c.cols;

        // Weights on a spectrum of ones.
        const std::vector<float> w512 = filter_ones(c.rows, c.cols, c.params, SimdMode::AVX512);
        const std::vector<float> w2 = filter_ones(c.rows, c.cols, c.params, SimdMode::AVX2);
        for (size_t idx = 0; idx < total; ++idx) {
            EXPECT_NEAR(w512[idx], w2[idx], 1e-6f) << "weight mismatch at idx " << idx;
        }

        // Full complex data with random values.
        std::mt19937 rng(static_cast<unsigned>(total));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        basalt::MemoryArena arena1(total * sizeof(float) * 2 + 4096);
        basalt::MemoryArena arena2(total * sizeof(float) * 2 + 4096);
        ComplexSoA a512(arena1, total);
        ComplexSoA a2(arena2, total);
        for (size_t idx = 0; idx < total; ++idx) {
            a512.real[idx] = dist(rng);
            a512.imag[idx] = dist(rng);
            a2.real[idx] = a512.real[idx];
            a2.imag[idx] = a512.imag[idx];
        }
        apply_fk_filter(a512, c.rows, c.cols, c.params, SimdMode::AVX512);
        apply_fk_filter(a2, c.rows, c.cols, c.params, SimdMode::AVX2);
        for (size_t idx = 0; idx < total; ++idx) {
            EXPECT_NEAR(a512.real[idx], a2.real[idx], 1e-6f) << "real mismatch at idx " << idx;
            EXPECT_NEAR(a512.imag[idx], a2.imag[idx], 1e-6f) << "imag mismatch at idx " << idx;
        }
    }
}

TEST(FKFilterTest, AsymmetricBandWeightsAreHermitianIncludingNyquist) {
    // The Nyquist row and column (even sizes) are their own alias of both signs, so a
    // one-sided weight would break w(f, k) == w(-f, -k) there and make the inverse complex.
    constexpr size_t rows = 64;
    constexpr size_t cols = 128;
    const FKParams params = make_params(-700.0, 1900.0, 150.0);

    const std::vector<float> paths[] = {
        filter_ones(rows, cols, params, SimdMode::AVX2),
        filter_ones(rows, cols, params, SimdMode::Auto),
    };
    for (const std::vector<float>& out : paths) {
        double max_asym = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            const size_t mi = (rows - i) % rows;
            for (size_t j = 0; j < cols; ++j) {
                const size_t mj = (cols - j) % cols;
                max_asym = std::max(max_asym, std::abs(static_cast<double>(out[i * cols + j]) -
                                                       static_cast<double>(out[mi * cols + mj])));
            }
        }
        EXPECT_LT(max_asym, 1e-6) << "weight is not Hermitian-symmetric";
    }
}

TEST(FKFilterTest, NyquistColumnFollowsAmendedMeanNotLiteralWeight) {
    // Spec amendment (see fk_filter.hpp): a Nyquist bin takes the mean of the literal weights at
    // v and -v. For an asymmetric band this differs from the literal weight at v, so the test
    // checks both the amended value and the gap to the literal formula. A silent revert to
    // either behaviour fails here. Parameters match the 64 x 128 case in the review report.
    constexpr size_t rows = 64;
    constexpr size_t cols = 128;
    FKParams params = make_params(-700.0, 1900.0, 150.0);
    params.dx = 12.5;
    const double df = 1.0 / (static_cast<double>(rows) * params.dt);
    const double dk = 1.0 / (static_cast<double>(cols) * params.dx);
    const size_t nyquist_col = cols / 2;
    const double k_nyquist = wrapped_axis(nyquist_col, cols, dk);   // +k_N

    const std::vector<float> out = filter_ones(rows, cols, params, SimdMode::AVX2);

    double max_amended_err = 0.0;
    double max_literal_gap = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        const double f = wrapped_axis(i, rows, df);
        const double literal = reference_weight(f, k_nyquist, params.mute_vel_min,
                                                params.mute_vel_max, params.taper_width);
        const double mirror = reference_weight(-f, k_nyquist, params.mute_vel_min,
                                               params.mute_vel_max, params.taper_width);
        const double amended = 0.5 * (literal + mirror);
        const double got = static_cast<double>(out[i * cols + nyquist_col]);
        max_amended_err = std::max(max_amended_err, std::abs(got - amended));
        max_literal_gap = std::max(max_literal_gap, std::abs(got - literal));
    }
    EXPECT_LT(max_amended_err, 1e-4) << "Nyquist column does not match the amended mean weight";
    EXPECT_GT(max_literal_gap, 0.1) << "Nyquist column equals the literal weight; the amendment is lost";
}

TEST(FKFilterTest, AsymmetricBandKeepsRealGatherReal) {
    // Forward 2D DFT of a real gather, filter, inverse DFT. The output must stay real
    // for an asymmetric band, which includes the Nyquist row and column (even sizes).
    constexpr size_t rows = 16;   // samples / frequency axis
    constexpr size_t cols = 32;   // traces / wavenumber axis
    constexpr size_t total = rows * cols;
    const double pi = 3.14159265358979323846;
    const FKParams params = make_params(-700.0, 1900.0, 150.0);

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> gather(total);                    // gather[x * rows + t]
    for (double& value : gather) {
        value = dist(rng);
    }

    // Naive forward DFT: X(i, j) = sum_x sum_t d(x, t) exp(-2 pi i (i t / rows + j x / cols)).
    std::vector<std::complex<double>> spectrum(total);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::complex<double> acc(0.0, 0.0);
            for (size_t x = 0; x < cols; ++x) {
                for (size_t t = 0; t < rows; ++t) {
                    const double phase = -2.0 * pi * (static_cast<double>(i * t) / static_cast<double>(rows) +
                                                      static_cast<double>(j * x) / static_cast<double>(cols));
                    acc += gather[x * rows + t] * std::complex<double>(std::cos(phase), std::sin(phase));
                }
            }
            spectrum[i * cols + j] = acc;
        }
    }

    basalt::MemoryArena arena(total * sizeof(float) * 2 + 4096);
    ComplexSoA data(arena, total);
    for (size_t idx = 0; idx < total; ++idx) {
        data.real[idx] = static_cast<float>(spectrum[idx].real());
        data.imag[idx] = static_cast<float>(spectrum[idx].imag());
    }
    apply_fk_filter(data, rows, cols, params, SimdMode::AVX2);

    // The band must actually remove something, otherwise the test proves nothing.
    double max_change = 0.0;
    for (size_t idx = 0; idx < total; ++idx) {
        const std::complex<double> filtered(data.real[idx], data.imag[idx]);
        max_change = std::max(max_change, std::abs(filtered - spectrum[idx]));
    }
    EXPECT_GT(max_change, 1.0);

    // Naive inverse DFT, d(x, t) = (1 / total) sum_i sum_j X'(i, j) exp(+2 pi i (i t / rows + j x / cols)).
    double max_re = 0.0;
    double max_im = 0.0;
    for (size_t x = 0; x < cols; ++x) {
        for (size_t t = 0; t < rows; ++t) {
            std::complex<double> acc(0.0, 0.0);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    const double phase = 2.0 * pi * (static_cast<double>(i * t) / static_cast<double>(rows) +
                                                     static_cast<double>(j * x) / static_cast<double>(cols));
                    const std::complex<double> filtered(data.real[i * cols + j], data.imag[i * cols + j]);
                    acc += filtered * std::complex<double>(std::cos(phase), std::sin(phase));
                }
            }
            acc /= static_cast<double>(total);
            max_re = std::max(max_re, std::abs(acc.real()));
            max_im = std::max(max_im, std::abs(acc.imag()));
        }
    }
    EXPECT_GT(max_re, 0.1);
    EXPECT_LT(max_im, 1e-4 * std::max(1.0, max_re)) << "filtered real gather has an imaginary part";
}
