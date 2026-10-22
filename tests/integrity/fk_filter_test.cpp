#include <gtest/gtest.h>
#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/internal/simd_dispatch.hpp"
#include "basalt/kernel/cpu_features.hpp"
#include "basalt/utils/fast_math.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <limits>
#include <stdexcept>

using namespace basalt::filter;
using namespace basalt::kernel;

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

TEST(FKFilterTest, MutesLowVelocity) {
    size_t rows = 256;
    size_t cols = 256;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(rows * cols));
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
    size_t rows = 16;
    size_t cols = 16;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(rows * cols));
    ComplexSoA data(arena, rows * cols);
    std::fill_n(data.real, rows * cols, 0.0f);
    std::fill_n(data.imag, rows * cols, 0.0f);

    // Set DC
    data.real[0] = 100.0f;
    data.imag[0] = -3.0f;

    FKParams params;
    apply_fk_filter(data, rows, cols, params);

    // DC (f=0, k=0) has infinite apparent velocity and passes unchanged.
    EXPECT_FLOAT_EQ(data.real[0], 100.0f);
    EXPECT_FLOAT_EQ(data.imag[0], -3.0f);
}

namespace {

// ---------------------------------------------------------------------------
// Double precision scalar reference of the documented filter semantics.
// ---------------------------------------------------------------------------

double ref_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double ref_band(double v, const FKParams& p) {
    const double s = 10.0 / p.taper_width;
    return 1.0 - ref_sigmoid((v - p.mute_vel_min) * s) * ref_sigmoid((p.mute_vel_max - v) * s);
}

double ref_signed_index(size_t index, size_t size, double step) {
    if (index <= size / 2) return static_cast<double>(index) * step;
    return (static_cast<double>(index) - static_cast<double>(size)) * step;
}

bool ref_nyquist(size_t index, size_t size) { return size % 2 == 0 && index == size / 2; }

double ref_weight(size_t i, size_t j, size_t rows, size_t cols, const FKParams& p) {
    const double f = ref_signed_index(i, rows, 1.0 / (static_cast<double>(rows) * p.dt));
    const double k = ref_signed_index(j, cols, 1.0 / (static_cast<double>(cols) * p.dx));
    if (k == 0.0) return 1.0;  // infinite apparent velocity (includes DC)
    const double v = f / k;
    // The Nyquist bin has no defined sign; the weight is symmetrised there.
    if (ref_nyquist(i, rows) || ref_nyquist(j, cols)) {
        return 0.5 * (ref_band(v, p) + ref_band(-v, p));
    }
    return ref_band(v, p);
}

struct Shape { size_t rows; size_t cols; };

const Shape kShapes[] = {{16, 17}, {64, 100}, {33, 7}, {300, 129}, {8, 1}, {16, 16}, {2, 2}};

// Deterministic pseudo-random fill in [-1, 1).
void fill_pattern(ComplexSoA& data, size_t n) {
    uint32_t state = 12345u;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / static_cast<float>(1u << 23) - 1.0f;
    };
    for (size_t i = 0; i < n; ++i) {
        data.real[i] = next();
        data.imag[i] = next();
    }
}

// True when the AVX-512 filter will really run (CPU support, BASALT_SIMD_MAX).
bool avx512_available() {
    return basalt::kernel::cpu_features().can_use_avx512();
}

struct Result {
    std::vector<float> re;
    std::vector<float> im;
};

Result run_filter(size_t rows, size_t cols, const FKParams& params, SimdMode mode) {
    const size_t n = rows * cols;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(n));
    ComplexSoA data(arena, n);
    fill_pattern(data, n);
    apply_fk_filter(data, rows, cols, params, mode);
    Result r;
    r.re.assign(data.real, data.real + n);
    r.im.assign(data.imag, data.imag + n);
    return r;
}

// Weight map: filter an all-ones real array and read the weights back.
std::vector<float> weight_map(size_t rows, size_t cols, const FKParams& params,
                              SimdMode mode = SimdMode::Auto) {
    const size_t n = rows * cols;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(n));
    ComplexSoA data(arena, n);
    std::fill_n(data.real, n, 1.0f);
    std::fill_n(data.imag, n, 0.0f);
    apply_fk_filter(data, rows, cols, params, mode);
    return std::vector<float>(data.real, data.real + n);
}

FKParams asymmetric_params() {
    FKParams p;
    p.mute_vel_min = -500.0;
    p.mute_vel_max = 3000.0;
    p.taper_width = 200.0;
    return p;
}

void expect_matches_reference(size_t rows, size_t cols, const FKParams& params, SimdMode mode,
                              const char* label) {
    const Result out = run_filter(rows, cols, params, mode);
    const size_t n = rows * cols;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(n));
    ComplexSoA orig(arena, n);
    fill_pattern(orig, n);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            const double w = ref_weight(i, j, rows, cols, params);
            ASSERT_NEAR(static_cast<double>(out.re[idx]), static_cast<double>(orig.real[idx]) * w, 1e-4)
                << label << " " << rows << "x" << cols << " i=" << i << " j=" << j;
            ASSERT_NEAR(static_cast<double>(out.im[idx]), static_cast<double>(orig.imag[idx]) * w, 1e-4)
                << label << " " << rows << "x" << cols << " i=" << i << " j=" << j;
        }
    }
}

} // namespace

TEST(FKFilterTest, DcPassesAndFZeroKNonZeroIsMuted) {
    const size_t rows = 16, cols = 16;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(rows * cols));
    ComplexSoA data(arena, rows * cols);
    std::fill_n(data.real, rows * cols, 1.0f);
    std::fill_n(data.imag, rows * cols, 0.0f);
    FKParams params;  // band [-1500, 1500] contains v = 0
    apply_fk_filter(data, rows, cols, params);

    EXPECT_FLOAT_EQ(data.real[0], 1.0f);                  // DC
    EXPECT_NEAR(data.real[0 * cols + 1], 0.0f, 1e-4f);    // f=0, k!=0 -> v=0, muted
    EXPECT_NEAR(data.real[0 * cols + 15], 0.0f, 1e-4f);   // f=0, k<0
    EXPECT_FLOAT_EQ(data.real[3 * cols + 0], 1.0f);       // k=0, f!=0 -> passes
    EXPECT_FLOAT_EQ(data.real[13 * cols + 0], 1.0f);      // k=0, f<0 -> passes
}

TEST(FKFilterTest, MatchesDoubleReferenceAllShapesAndModes) {
    FKParams sym;
    const FKParams asym = asymmetric_params();
    for (const Shape& sh : kShapes) {
        expect_matches_reference(sh.rows, sh.cols, sym, SimdMode::Scalar, "scalar/sym");
        expect_matches_reference(sh.rows, sh.cols, asym, SimdMode::Scalar, "scalar/asym");
        expect_matches_reference(sh.rows, sh.cols, sym, SimdMode::AVX2, "avx2/sym");
        expect_matches_reference(sh.rows, sh.cols, asym, SimdMode::AVX2, "avx2/asym");
        expect_matches_reference(sh.rows, sh.cols, asym, SimdMode::Auto, "auto/asym");
        if (avx512_available()) {
            expect_matches_reference(sh.rows, sh.cols, sym, SimdMode::AVX512, "avx512/sym");
            expect_matches_reference(sh.rows, sh.cols, asym, SimdMode::AVX512, "avx512/asym");
        }
    }
}

TEST(FKFilterTest, Avx2AndAvx512Agree) {
    if (!avx512_available()) {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }
    const FKParams asym = asymmetric_params();
    const FKParams sym;
    for (const Shape& sh : kShapes) {
        for (const FKParams& p : {sym, asym}) {
            const Result a = run_filter(sh.rows, sh.cols, p, SimdMode::AVX2);
            const Result b = run_filter(sh.rows, sh.cols, p, SimdMode::AVX512);
            ASSERT_EQ(a.re.size(), b.re.size());
            for (size_t i = 0; i < a.re.size(); ++i) {
                ASSERT_NEAR(a.re[i], b.re[i], 1e-6f) << sh.rows << "x" << sh.cols << " idx " << i;
                ASSERT_NEAR(a.im[i], b.im[i], 1e-6f) << sh.rows << "x" << sh.cols << " idx " << i;
            }
        }
    }
}

TEST(FKFilterTest, HermitianSymmetryHoldsExactly) {
    const FKParams sym;
    const FKParams asym = asymmetric_params();
    for (SimdMode mode : {SimdMode::Scalar, SimdMode::AVX2, SimdMode::AVX512}) {
        for (const Shape& sh : kShapes) {
            for (const FKParams& p : {sym, asym}) {
                const std::vector<float> w = weight_map(sh.rows, sh.cols, p, mode);
                for (size_t i = 0; i < sh.rows; ++i) {
                    for (size_t j = 0; j < sh.cols; ++j) {
                        const size_t pi = (sh.rows - i) % sh.rows;
                        const size_t pj = (sh.cols - j) % sh.cols;
                        ASSERT_EQ(w[i * sh.cols + j], w[pi * sh.cols + pj])
                            << sh.rows << "x" << sh.cols << " (" << i << "," << j << ") vs ("
                            << pi << "," << pj << ")";
                    }
                }
            }
        }
    }
}

TEST(FKFilterTest, AsymmetricBandHonoursSignOfVelocity) {
    // 64x64, dt=0.004, dx=25: v = (i/j) * 6250 for the first quadrant.
    FKParams p;
    p.mute_vel_min = 1000.0;
    p.mute_vel_max = 2000.0;
    p.taper_width = 50.0;
    const size_t n = 64;
    const std::vector<float> w = weight_map(n, n, p);
    auto at = [&](size_t i, size_t j) { return w[i * n + j]; };

    // (f>0, k>0): v = +1562 -> inside the band, muted
    EXPECT_NEAR(at(1, 4), 0.0f, 1e-3f);
    // (f>0, k<0): v = -1562 -> outside the band, passes
    EXPECT_NEAR(at(1, n - 4), 1.0f, 1e-3f);
    // (f<0, k>0): v = -1562 -> passes
    EXPECT_NEAR(at(n - 1, 4), 1.0f, 1e-3f);
    // (f<0, k<0): v = +1562 -> muted
    EXPECT_NEAR(at(n - 1, n - 4), 0.0f, 1e-3f);
    // Positive velocity above the band: v = 6250 passes
    EXPECT_NEAR(at(1, 1), 1.0f, 1e-3f);
}

TEST(FKFilterTest, SymmetricBandMatchesAbsoluteVelocityFormula) {
    // With min == -max the signed band must reproduce the classic
    // sigmoid(|v| - vmax) mute.
    const FKParams p;  // [-1500, 1500], taper 100
    const size_t rows = 64, cols = 100;
    const std::vector<float> w = weight_map(rows, cols, p);
    const double s = 10.0 / p.taper_width;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 1; j < cols; ++j) {  // skip k = 0
            const double f = ref_signed_index(i, rows, 1.0 / (static_cast<double>(rows) * p.dt));
            const double k = ref_signed_index(j, cols, 1.0 / (static_cast<double>(cols) * p.dx));
            const double expected = ref_sigmoid((std::fabs(f / k) - p.mute_vel_max) * s);
            ASSERT_NEAR(static_cast<double>(w[i * cols + j]), expected, 1e-4) << "i=" << i << " j=" << j;
        }
    }
}

TEST(FKFilterTest, NoEpsilonSkewAtLargeGrids) {
    // The old k + 1e-6 offset skewed weights by up to ~0.058 at cols=4096,
    // dx=500 (dk ~ 4.9e-7). The result must now match the exact reference.
    FKParams p;
    p.dx = 500.0;
    const size_t rows = 4, cols = 4096;
    const std::vector<float> w = weight_map(rows, cols, p);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ASSERT_NEAR(static_cast<double>(w[i * cols + j]), ref_weight(i, j, rows, cols, p), 1e-4)
                << "i=" << i << " j=" << j;
        }
    }
}

TEST(FKFilterTest, ParameterValidationThrows) {
    const size_t rows = 8, cols = 8;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(rows * cols));
    ComplexSoA data(arena, rows * cols);
    std::fill_n(data.real, rows * cols, 1.0f);
    std::fill_n(data.imag, rows * cols, 1.0f);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    auto expect_throw = [&](const FKParams& p, const char* what) {
        EXPECT_THROW(apply_fk_filter(data, rows, cols, p), std::invalid_argument) << what;
        EXPECT_THROW(apply_fk_filter(data, rows, cols, p, SimdMode::AVX2), std::invalid_argument)
            << what;
        EXPECT_THROW(apply_fk_filter(data, rows, cols, p, SimdMode::Scalar), std::invalid_argument)
            << what;
    };
    FKParams p;
    p = FKParams{}; p.taper_width = nan; expect_throw(p, "taper NaN");
    p = FKParams{}; p.taper_width = 0.0; expect_throw(p, "taper 0");
    p = FKParams{}; p.taper_width = -1.0; expect_throw(p, "taper negative");
    p = FKParams{}; p.taper_width = inf; expect_throw(p, "taper inf");
    p = FKParams{}; p.dt = 0.0; expect_throw(p, "dt 0");
    p = FKParams{}; p.dt = -0.004; expect_throw(p, "dt negative");
    p = FKParams{}; p.dt = nan; expect_throw(p, "dt NaN");
    p = FKParams{}; p.dx = 0.0; expect_throw(p, "dx 0");
    p = FKParams{}; p.dx = -25.0; expect_throw(p, "dx negative");
    p = FKParams{}; p.dx = inf; expect_throw(p, "dx inf");
    p = FKParams{}; p.mute_vel_min = 1500.0; p.mute_vel_max = 1500.0; expect_throw(p, "min == max");
    p = FKParams{}; p.mute_vel_min = 2000.0; p.mute_vel_max = 1500.0; expect_throw(p, "min > max");
    p = FKParams{}; p.mute_vel_min = nan; expect_throw(p, "min NaN");
    p = FKParams{}; p.mute_vel_max = inf; expect_throw(p, "max inf");

    // Data must not have been touched by rejected calls.
    for (size_t i = 0; i < rows * cols; ++i) {
        ASSERT_EQ(data.real[i], 1.0f);
        ASSERT_EQ(data.imag[i], 1.0f);
    }
}

TEST(FKFilterTest, BufferValidationThrows) {
    const size_t rows = 8, cols = 8;
    basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(rows * cols));
    ComplexSoA data(arena, rows * cols);
    std::fill_n(data.real, rows * cols, 1.0f);
    std::fill_n(data.imag, rows * cols, 1.0f);
    const FKParams p;

    // rows * cols larger than the buffer
    EXPECT_THROW(apply_fk_filter(data, rows + 1, cols, p), std::invalid_argument);
    EXPECT_THROW(apply_fk_filter(data, rows, cols + 1, p, SimdMode::AVX2), std::invalid_argument);
    EXPECT_THROW(apply_fk_filter(data, rows, cols + 1, p, SimdMode::Scalar), std::invalid_argument);
    // rows * cols overflow
    EXPECT_THROW(apply_fk_filter(data, std::numeric_limits<size_t>::max() / 2 + 1, 2, p),
                 std::invalid_argument);
    // null buffers
    ComplexSoA empty;
    EXPECT_THROW(apply_fk_filter(empty, rows, cols, p), std::invalid_argument);
    // Empty grids are a harmless no-op, even with null buffers
    EXPECT_NO_THROW(apply_fk_filter(empty, 0, cols, p));
    EXPECT_NO_THROW(apply_fk_filter(empty, rows, 0, p));
}

namespace {

// fast_math.hpp helpers require AVX2+FMA, so the probes carry the target
// attribute and the test only calls them when the CPU supports it.
BASALT_TARGET_AVX2 void probe_fast_exp(const float* in, float* out) {
    _mm256_storeu_ps(out, basalt::utils::fast_exp_avx(_mm256_loadu_ps(in)));
}

BASALT_TARGET_AVX2 void probe_fast_sigmoid(const float* in, float* out) {
    _mm256_storeu_ps(out, basalt::utils::fast_sigmoid_avx(_mm256_loadu_ps(in)));
}

} // namespace

TEST(FKFilterTest, FastMathClampsAndPropagatesNaN) {
    if (!basalt::kernel::cpu_features().can_use_avx2()) {
        GTEST_SKIP() << "AVX2/FMA not available (or capped by BASALT_SIMD_MAX)";
    }
    alignas(32) float in[8] = {-100.0f, -88.0f, -87.0f, 0.0f, 87.0f, 88.0f, 100.0f, 1e30f};
    alignas(32) float out[8];
    probe_fast_exp(in, out);
    for (int i = 0; i < 8; ++i) {
        EXPECT_GT(out[i], 0.0f) << "exp(" << in[i] << ")";
        EXPECT_TRUE(std::isfinite(out[i])) << "exp(" << in[i] << ")";
    }
    EXPECT_NEAR(out[0] / out[2], 1.0f, 1e-3f);          // exp(-100) saturates at exp(-87)
    EXPECT_NEAR(out[3], 1.0f, 1e-5f);
    EXPECT_NEAR(out[6] / out[5], 1.0f, 1e-3f);          // saturates at exp(88)

    alignas(32) float nan_in[8] = {std::nanf(""), -1.0f, 1.0f, std::nanf(""), 0.0f, 0.0f, 0.0f, 0.0f};
    probe_fast_sigmoid(nan_in, out);
    EXPECT_TRUE(std::isnan(out[0]));
    EXPECT_TRUE(std::isnan(out[3]));
    EXPECT_NEAR(out[1], 1.0f / (1.0f + std::exp(1.0f)), 1e-5f);
    EXPECT_NEAR(out[2], 1.0f / (1.0f + std::exp(-1.0f)), 1e-5f);
    probe_fast_exp(nan_in, out);
    EXPECT_TRUE(std::isnan(out[0]));
}

// ---------------------------------------------------------------------------
// Scalar path and SIMD dispatch
// ---------------------------------------------------------------------------

TEST(FKFilterScalarTest, WeightsMatchAvx2Within1e5) {
    const FKParams sym;
    const FKParams asym = asymmetric_params();
    FKParams narrow = asym;       // sharp taper stresses the sigmoid slope
    narrow.taper_width = 10.0;
    for (const Shape& sh : kShapes) {
        for (const FKParams& p : {sym, asym, narrow}) {
            const std::vector<float> ws = weight_map(sh.rows, sh.cols, p, SimdMode::Scalar);
            const std::vector<float> wa = weight_map(sh.rows, sh.cols, p, SimdMode::AVX2);
            ASSERT_EQ(ws.size(), wa.size());
            for (size_t i = 0; i < ws.size(); ++i) {
                ASSERT_NEAR(ws[i], wa[i], 1e-5f) << sh.rows << "x" << sh.cols << " idx " << i;
            }
        }
    }
}

TEST(FKFilterScalarTest, WeightsMatchDoubleReferenceTightly) {
    // The scalar path uses std::exp, so it should track the double reference
    // closely (float rounding of the velocity only).
    const FKParams asym = asymmetric_params();
    for (const Shape& sh : kShapes) {
        const std::vector<float> w = weight_map(sh.rows, sh.cols, asym, SimdMode::Scalar);
        for (size_t i = 0; i < sh.rows; ++i) {
            for (size_t j = 0; j < sh.cols; ++j) {
                ASSERT_NEAR(static_cast<double>(w[i * sh.cols + j]),
                            ref_weight(i, j, sh.rows, sh.cols, asym), 1e-5)
                    << sh.rows << "x" << sh.cols << " i=" << i << " j=" << j;
            }
        }
    }
}

TEST(FKFilterScalarTest, PassesKZeroAndDcAndSymmetrisesNyquist) {
    const size_t rows = 16, cols = 16;
    const FKParams asym = asymmetric_params();
    const std::vector<float> w = weight_map(rows, cols, asym, SimdMode::Scalar);
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_EQ(w[i * cols + 0], 1.0f) << "k == 0 must pass, row " << i;
    }
    // Nyquist column: weight equals the average over +v and -v, so the weight
    // at (i, cols/2) equals the weight at the mirrored row (rows - i, cols/2).
    for (size_t i = 1; i < rows; ++i) {
        EXPECT_EQ(w[i * cols + cols / 2], w[(rows - i) * cols + cols / 2]) << "row " << i;
    }
}

TEST(FKFilterScalarTest, ScalarOutputIsFiniteAndPreservesDataLayout) {
    // Non-square, odd sizes: only the weight multiplies the data, the pattern
    // survives scaled (real and imag use the same weight).
    const size_t rows = 33, cols = 7;
    const FKParams p;
    const Result before = [&] {
        basalt::MemoryArena arena(ComplexSoA::required_arena_bytes(rows * cols));
        ComplexSoA d(arena, rows * cols);
        fill_pattern(d, rows * cols);
        return Result{std::vector<float>(d.real, d.real + rows * cols),
                      std::vector<float>(d.imag, d.imag + rows * cols)};
    }();
    const Result after = run_filter(rows, cols, p, SimdMode::Scalar);
    const std::vector<float> w = weight_map(rows, cols, p, SimdMode::Scalar);
    for (size_t i = 0; i < rows * cols; ++i) {
        ASSERT_TRUE(std::isfinite(after.re[i]) && std::isfinite(after.im[i]));
        ASSERT_FLOAT_EQ(after.re[i], before.re[i] * w[i]);
        ASSERT_FLOAT_EQ(after.im[i], before.im[i] * w[i]);
    }
}

TEST(FKFilterDispatchTest, CapsSelectPathsAndFallBack) {
    using basalt::kernel::detail::ScopedSimdCap;
    const FKParams asym = asymmetric_params();
    const Shape sh{64, 100};

    const std::vector<float> scalar = weight_map(sh.rows, sh.cols, asym, SimdMode::Scalar);
    const std::vector<float> avx2 = weight_map(sh.rows, sh.cols, asym, SimdMode::AVX2);
    {
        // Capped to scalar: every request (even AVX512) must produce the scalar result.
        ScopedSimdCap cap(SimdMode::Scalar);
        for (SimdMode mode : {SimdMode::Auto, SimdMode::AVX2, SimdMode::AVX512, SimdMode::Scalar}) {
            EXPECT_EQ(weight_map(sh.rows, sh.cols, asym, mode), scalar)
                << "mode " << static_cast<int>(mode);
        }
    }
    {
        // Capped to AVX2: Auto and AVX512 requests must run the AVX2 path.
        ScopedSimdCap cap(SimdMode::AVX2);
        for (SimdMode mode : {SimdMode::Auto, SimdMode::AVX2, SimdMode::AVX512}) {
            EXPECT_EQ(weight_map(sh.rows, sh.cols, asym, mode), avx2)
                << "mode " << static_cast<int>(mode);
        }
        EXPECT_EQ(weight_map(sh.rows, sh.cols, asym, SimdMode::Scalar), scalar);
    }
    // Explicit AVX2 / Auto without a cap on an AVX2 machine must be the AVX2 path
    // (or scalar on a CPU without it): never garbage.
    const std::vector<float> autow = weight_map(sh.rows, sh.cols, asym, SimdMode::Auto);
    ASSERT_EQ(autow.size(), scalar.size());
    for (size_t i = 0; i < autow.size(); ++i) {
        ASSERT_NEAR(autow[i], scalar[i], 1e-5f) << i;
    }
}

TEST(FKFilterDispatchTest, EveryModeMatchesReferenceUnderEveryCap) {
    using basalt::kernel::detail::ScopedSimdCap;
    const FKParams asym = asymmetric_params();
    for (SimdMode cap_mode : {SimdMode::Scalar, SimdMode::AVX2, SimdMode::AVX512}) {
        ScopedSimdCap cap(cap_mode);
        for (SimdMode mode : {SimdMode::Auto, SimdMode::Scalar, SimdMode::AVX2, SimdMode::AVX512}) {
            expect_matches_reference(64, 100, asym, mode, "capped");
            expect_matches_reference(33, 7, asym, mode, "capped");
        }
    }
}
