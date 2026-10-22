#include "basalt/arena.hpp"
#include "basalt/internal/fft2d_auto.hpp"
#include "basalt/internal/simd_dispatch.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/fft2d.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace basalt::kernel;

namespace {

// ---------- Double-precision reference ----------

/// 1D DFT of 'count' lines in double precision. Lines are 'len' long and
/// 'stride' apart; the first element of line l is at offset l * line_step.
void reference_dft_lines(std::vector<double>& re, std::vector<double>& im, size_t len,
                         size_t stride, size_t count, size_t line_step) {
    std::vector<double> cs(len), sn(len);
    for (size_t m = 0; m < len; ++m) {
        const double a = -2.0 * M_PI * static_cast<double>(m) / static_cast<double>(len);
        cs[m] = std::cos(a);
        sn[m] = std::sin(a);
    }
    std::vector<double> tr(len), ti(len);
    for (size_t l = 0; l < count; ++l) {
        const size_t base = l * line_step;
        for (size_t k = 0; k < len; ++k) {
            double sr = 0.0, si = 0.0;
            size_t idx = 0;
            for (size_t j = 0; j < len; ++j) {
                const double xr = re[base + j * stride];
                const double xi = im[base + j * stride];
                sr += xr * cs[idx] - xi * sn[idx];
                si += xr * sn[idx] + xi * cs[idx];
                idx += k;
                if (idx >= len)
                    idx -= len;
            }
            tr[k] = sr;
            ti[k] = si;
        }
        for (size_t k = 0; k < len; ++k) {
            re[base + k * stride] = tr[k];
            im[base + k * stride] = ti[k];
        }
    }
}

/// Forward 2D DFT of a row-major rows x cols array, in double precision.
void reference_dft2d(const std::vector<float>& in_re, const std::vector<float>& in_im, size_t rows,
                     size_t cols, std::vector<double>& re, std::vector<double>& im) {
    re.assign(in_re.begin(), in_re.end());
    im.assign(in_im.begin(), in_im.end());
    reference_dft_lines(re, im, cols, 1, rows, cols); // along rows
    reference_dft_lines(re, im, rows, cols, cols, 1); // along columns
}

double rel_l2_error(const std::vector<float>& got_re, const std::vector<float>& got_im,
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

double rel_l2_error(const std::vector<float>& got_re, const std::vector<float>& got_im,
                    const std::vector<float>& ref_re, const std::vector<float>& ref_im) {
    return rel_l2_error(got_re, got_im, std::vector<double>(ref_re.begin(), ref_re.end()),
                        std::vector<double>(ref_im.begin(), ref_im.end()));
}

/// c * log2(total) * FLT_EPSILON, the same shape of bound as the 1D tests.
double fft2d_error_bound(size_t rows, size_t cols) {
    constexpr double c = 4.0;
    const double total = static_cast<double>(rows) * static_cast<double>(cols);
    return c * std::max(1.0, std::log2(total)) * static_cast<double>(FLT_EPSILON);
}

void fill_random(std::vector<float>& re, std::vector<float>& im, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : re)
        v = dist(rng);
    for (auto& v : im)
        v = dist(rng);
}

const char* schedule_name(FFT2DSchedule s) {
    switch (s) {
    case FFT2DSchedule::Auto:
        return "Auto";
    case FFT2DSchedule::LegacyColumnFirst:
        return "Legacy";
    case FFT2DSchedule::FourStepRowFirst:
        return "FourStep";
    case FFT2DSchedule::RowFirstColumnStrips:
        return "Strips";
    }
    return "?";
}

/// exp(+2*pi*i*(k0*r/rows + l0*c/cols)) as float SoA.
void fill_plane_wave(std::vector<float>& re, std::vector<float>& im, size_t rows, size_t cols,
                     size_t k0, size_t l0) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const double phase = static_cast<double>((k0 * r) % rows) / static_cast<double>(rows) +
                                 static_cast<double>((l0 * c) % cols) / static_cast<double>(cols);
            const double a = 2.0 * M_PI * phase;
            re[r * cols + c] = static_cast<float>(std::cos(a));
            im[r * cols + c] = static_cast<float>(std::sin(a));
        }
    }
}

} // namespace

// ---------- Accuracy against the double-precision reference ----------

struct Shape2D {
    size_t rows;
    size_t cols;
};

using AccuracyParam = std::tuple<Shape2D, FFT2DSchedule, size_t>;

class FFT2DAccuracyTest : public ::testing::TestWithParam<AccuracyParam> {};

TEST_P(FFT2DAccuracyTest, ForwardMatchesReferenceAndRoundTrips) {
    const Shape2D shape = std::get<0>(GetParam());
    const FFT2DConfig config{std::get<1>(GetParam()), std::get<2>(GetParam())};
    const size_t rows = shape.rows, cols = shape.cols, total = rows * cols;

    std::vector<float> in_re(total), in_im(total);
    fill_random(in_re, in_im, static_cast<uint32_t>(rows * 131 + cols));

    std::vector<double> ref_re, ref_im;
    reference_dft2d(in_re, in_im, rows, cols, ref_re, ref_im);

    std::vector<float> re = in_re, im = in_im;
    // Scratch sized exactly by the public helper, never reset by the test.
    basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols, config));

    fft2d_forward(re.data(), im.data(), rows, cols, scratch, config);
    const double fwd_err = rel_l2_error(re, im, ref_re, ref_im);
    EXPECT_LE(fwd_err, fft2d_error_bound(rows, cols))
        << rows << "x" << cols << " " << schedule_name(config.schedule) << " tile "
        << config.transpose_tile << ": forward rel L2 error " << fwd_err;

    fft2d_inverse(re.data(), im.data(), rows, cols, scratch, config);
    const double rt_err = rel_l2_error(re, im, in_re, in_im);
    EXPECT_LE(rt_err, 2.0 * fft2d_error_bound(rows, cols))
        << rows << "x" << cols << " " << schedule_name(config.schedule) << " tile "
        << config.transpose_tile << ": round-trip rel L2 error " << rt_err;
}

INSTANTIATE_TEST_SUITE_P(
    FFT2DReference, FFT2DAccuracyTest,
    ::testing::Combine(::testing::Values(Shape2D{1, 64}, Shape2D{64, 1}, Shape2D{2, 2},
                                         Shape2D{8, 8}, Shape2D{16, 32}, Shape2D{8, 256},
                                         Shape2D{256, 8}, Shape2D{128, 512}, Shape2D{256, 256}),
                       ::testing::Values(FFT2DSchedule::LegacyColumnFirst,
                                         FFT2DSchedule::FourStepRowFirst,
                                         FFT2DSchedule::RowFirstColumnStrips, FFT2DSchedule::Auto),
                       ::testing::Values(size_t{96}, size_t{13}, size_t{128})),
    [](const ::testing::TestParamInfo<AccuracyParam>& param_info) {
        const Shape2D s = std::get<0>(param_info.param);
        return std::to_string(s.rows) + "x" + std::to_string(s.cols) + "_" +
               schedule_name(std::get<1>(param_info.param)) + "_tile" +
               std::to_string(std::get<2>(param_info.param));
    });

// ---------- Sign convention ----------

TEST(FFT2DTest, ForwardOfPositivePlaneWaveLandsInBinK0L0) {
    // exp(+2*pi*i*(k0*r/R + l0*c/C)) -> all energy at (k0, l0), for every schedule.
    constexpr size_t rows = 32, cols = 64, total = rows * cols;
    constexpr size_t k0 = 3, l0 = 5;
    for (FFT2DSchedule sched : {FFT2DSchedule::LegacyColumnFirst, FFT2DSchedule::FourStepRowFirst,
                                FFT2DSchedule::RowFirstColumnStrips, FFT2DSchedule::Auto}) {
        std::vector<float> re(total), im(total);
        fill_plane_wave(re, im, rows, cols, k0, l0);
        basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));

        fft2d_forward(re.data(), im.data(), rows, cols, scratch, FFT2DConfig{sched, 96});

        const float tol = 2e-3f * static_cast<float>(total);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                const float mag = std::hypot(re[r * cols + c], im[r * cols + c]);
                if (r == k0 && c == l0) {
                    EXPECT_NEAR(mag, static_cast<float>(total), tol) << schedule_name(sched);
                } else {
                    ASSERT_LE(mag, tol) << schedule_name(sched) << " leakage at " << r << "," << c;
                }
            }
        }
    }
}

TEST(FFT2DTest, InverseOfKnownSpectrum) {
    // A single spectral bin X[k0,l0] = a+bi inverts to (a+bi)/(R*C) * plane wave.
    // Also pins the single 1/(rows*cols) scaling of the inverse.
    constexpr size_t rows = 16, cols = 32, total = rows * cols;
    constexpr size_t k0 = 2, l0 = 7;
    for (FFT2DSchedule sched : {FFT2DSchedule::LegacyColumnFirst, FFT2DSchedule::FourStepRowFirst,
                                FFT2DSchedule::RowFirstColumnStrips}) {
        std::vector<float> re(total, 0.0f), im(total, 0.0f);
        re[k0 * cols + l0] = 2.0f;
        im[k0 * cols + l0] = -1.0f;
        basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));

        fft2d_inverse(re.data(), im.data(), rows, cols, scratch, FFT2DConfig{sched, 128});

        std::vector<float> wr(total), wi(total);
        fill_plane_wave(wr, wi, rows, cols, k0, l0);
        for (size_t i = 0; i < total; ++i) {
            const double er = (2.0 * wr[i] + 1.0 * wi[i]) / static_cast<double>(total);
            const double ei = (2.0 * wi[i] - 1.0 * wr[i]) / static_cast<double>(total);
            ASSERT_NEAR(re[i], er, 1e-6) << schedule_name(sched) << " real at " << i;
            ASSERT_NEAR(im[i], ei, 1e-6) << schedule_name(sched) << " imag at " << i;
        }
    }
}

TEST(FFT2DTest, InverseOfFlatSpectrumIsDelta) {
    constexpr size_t rows = 8, cols = 16, total = rows * cols;
    std::vector<float> re(total, 1.0f), im(total, 0.0f);
    basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));
    fft2d_inverse(re.data(), im.data(), rows, cols, scratch);
    EXPECT_NEAR(re[0], 1.0f, 1e-6f);
    for (size_t i = 1; i < total; ++i) {
        EXPECT_NEAR(re[i], 0.0f, 1e-6f) << i;
    }
    for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(im[i], 0.0f, 1e-6f) << i;
    }
}

// ---------- Every Auto branch, at real sizes ----------

TEST(FFT2DAutoConfigTest, ChoosesExpectedConfigPerSizeClass) {
    using basalt::internal::default_fft2d_config;
    struct Case {
        size_t rows, cols;
        FFT2DSchedule sched;
        size_t tile;
    };
    // Every shape resolves to the transpose-free column-strip schedule.
    const Case cases[] = {
        {256, 256, FFT2DSchedule::RowFirstColumnStrips, 128},
        {512, 512, FFT2DSchedule::RowFirstColumnStrips, 128},
        {1024, 1024, FFT2DSchedule::RowFirstColumnStrips, 128},
        {2048, 2048, FFT2DSchedule::RowFirstColumnStrips, 128},
        {4096, 4096, FFT2DSchedule::RowFirstColumnStrips, 128},
        {8, 8, FFT2DSchedule::RowFirstColumnStrips, 128},
        {128, 512, FFT2DSchedule::RowFirstColumnStrips, 128},
        {4096, 1024, FFT2DSchedule::RowFirstColumnStrips, 128},
        {1024, 4096, FFT2DSchedule::RowFirstColumnStrips, 128},
        {8192, 512, FFT2DSchedule::RowFirstColumnStrips, 128},
        {1, 1 << 20, FFT2DSchedule::RowFirstColumnStrips, 128},
    };
    for (const Case& c : cases) {
        const FFT2DConfig cfg = default_fft2d_config(c.rows, c.cols);
        EXPECT_EQ(cfg.schedule, c.sched) << c.rows << "x" << c.cols;
        EXPECT_EQ(cfg.transpose_tile, c.tile) << c.rows << "x" << c.cols;
        // resolved_fft2d_config only replaces Auto.
        EXPECT_NE(basalt::internal::resolved_fft2d_config(FFT2DConfig{}, c.rows, c.cols).schedule,
                  FFT2DSchedule::Auto);
        const FFT2DConfig explicit_cfg{FFT2DSchedule::FourStepRowFirst, 7};
        const FFT2DConfig kept =
            basalt::internal::resolved_fft2d_config(explicit_cfg, c.rows, c.cols);
        EXPECT_EQ(kept.schedule, explicit_cfg.schedule);
        EXPECT_EQ(kept.transpose_tile, explicit_cfg.transpose_tile);
    }
}

class FFT2DAutoSizeTest : public ::testing::TestWithParam<Shape2D> {};

TEST_P(FFT2DAutoSizeTest, PlaneWavePeakAndRoundTrip) {
    // Large shapes are too big for the O(n^2) reference; a plane wave has a
    // closed-form spectrum (a single bin of height rows*cols), which also pins
    // the sign convention for the Auto-selected schedule.
    const size_t rows = GetParam().rows, cols = GetParam().cols, total = rows * cols;
    const size_t k0 = 3, l0 = 5;
    std::vector<float> re(total), im(total);
    fill_plane_wave(re, im, rows, cols, k0, l0);
    basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));

    fft2d_forward(re.data(), im.data(), rows, cols, scratch); // Auto

    double leak = 0.0;
    for (size_t i = 0; i < total; ++i) {
        if (i == k0 * cols + l0)
            continue;
        leak += static_cast<double>(re[i]) * re[i] + static_cast<double>(im[i]) * im[i];
    }
    const double peak = std::hypot(static_cast<double>(re[k0 * cols + l0]),
                                   static_cast<double>(im[k0 * cols + l0]));
    EXPECT_NEAR(peak, static_cast<double>(total), 1e-4 * static_cast<double>(total));
    EXPECT_LE(std::sqrt(leak), 4.0 * fft2d_error_bound(rows, cols) * static_cast<double>(total));

    fft2d_inverse(re.data(), im.data(), rows, cols, scratch); // Auto
    std::vector<float> wr(total), wi(total);
    fill_plane_wave(wr, wi, rows, cols, k0, l0);
    EXPECT_LE(rel_l2_error(re, im, wr, wi), 2.0 * fft2d_error_bound(rows, cols));
}

INSTANTIATE_TEST_SUITE_P(AutoBranches, FFT2DAutoSizeTest,
                         ::testing::Values(Shape2D{256, 256}, Shape2D{512, 512},
                                           Shape2D{1024, 1024}, Shape2D{2048, 2048},
                                           Shape2D{4096, 1024}, Shape2D{512, 1024},
                                           Shape2D{64, 4096}),
                         [](const ::testing::TestParamInfo<Shape2D>& param_info) {
                             return std::to_string(param_info.param.rows) + "x" +
                                    std::to_string(param_info.param.cols);
                         });

// ---------- Scratch handling ----------

TEST(FFT2DScratchTest, RepeatedCallsOnOneArenaNeedNoManualReset) {
    constexpr size_t rows = 32, cols = 64, total = rows * cols;
    std::vector<float> in_re(total), in_im(total);
    fill_random(in_re, in_im, 5u);
    std::vector<float> re = in_re, im = in_im;

    // Exactly the documented requirement; never reset by the caller.
    basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));
    for (int i = 0; i < 3; ++i) {
        fft2d_forward(re.data(), im.data(), rows, cols, scratch);
        EXPECT_EQ(scratch.bytes_used(), 0u) << "scratch not released after forward #" << i;
        fft2d_inverse(re.data(), im.data(), rows, cols, scratch);
        EXPECT_EQ(scratch.bytes_used(), 0u) << "scratch not released after inverse #" << i;
    }
    EXPECT_LE(rel_l2_error(re, im, in_re, in_im), 3.0 * fft2d_error_bound(rows, cols));
}

TEST(FFT2DScratchTest, TooSmallScratchThrowsBadAllocAndLeavesDataUntouched) {
    constexpr size_t rows = 32, cols = 32, total = rows * cols;
    std::vector<float> in_re(total), in_im(total);
    fill_random(in_re, in_im, 6u);
    std::vector<float> re = in_re, im = in_im;

    // Room for one plane but not two.
    basalt::MemoryArena small(total * sizeof(float) + 512);
    EXPECT_THROW(fft2d_forward(re.data(), im.data(), rows, cols, small), std::bad_alloc);
    EXPECT_THROW(fft2d_inverse(re.data(), im.data(), rows, cols, small), std::bad_alloc);
    EXPECT_EQ(re, in_re);
    EXPECT_EQ(im, in_im);
    EXPECT_EQ(small.bytes_used(), 0u);

    // Arena that cannot hold even one plane.
    basalt::MemoryArena tiny(64);
    EXPECT_THROW(fft2d_forward(re.data(), im.data(), rows, cols, tiny), std::bad_alloc);
    EXPECT_EQ(re, in_re);

    // Every explicit schedule fails before touching the data (the column-strip
    // schedule must not run its row pass before its buffers are secured).
    for (FFT2DSchedule sched : {FFT2DSchedule::LegacyColumnFirst, FFT2DSchedule::FourStepRowFirst,
                                FFT2DSchedule::RowFirstColumnStrips}) {
        EXPECT_THROW(fft2d_forward(re.data(), im.data(), rows, cols, tiny, FFT2DConfig{sched, 64}),
                     std::bad_alloc)
            << schedule_name(sched);
        EXPECT_THROW(fft2d_inverse(re.data(), im.data(), rows, cols, tiny, FFT2DConfig{sched, 64}),
                     std::bad_alloc)
            << schedule_name(sched);
        EXPECT_EQ(re, in_re) << schedule_name(sched);
        EXPECT_EQ(im, in_im) << schedule_name(sched);
        EXPECT_EQ(tiny.bytes_used(), 0u) << schedule_name(sched);
    }
}

TEST(FFT2DScratchTest, SharedArenaKeepsCallerAllocationsAndRewinds) {
    constexpr size_t rows = 16, cols = 16, total = rows * cols;
    std::vector<float> in_re(total), in_im(total);
    fill_random(in_re, in_im, 7u);

    // The arena already holds a caller allocation. fft2d must leave it intact
    // and hand its own scratch back on return, so it can be called repeatedly
    // with room for exactly one call's scratch.
    basalt::MemoryArena arena(fft2d_scratch_bytes(rows, cols) + 256);
    float* mine = arena.allocate_array<float>(8);
    ASSERT_NE(mine, nullptr);
    for (size_t i = 0; i < 8; ++i) {
        mine[i] = static_cast<float>(i);
    }
    const size_t used_before = arena.bytes_used();

    std::vector<float> re = in_re, im = in_im;
    fft2d_forward(re.data(), im.data(), rows, cols, arena);
    EXPECT_EQ(arena.bytes_used(), used_before);
    fft2d_inverse(re.data(), im.data(), rows, cols, arena);
    EXPECT_EQ(arena.bytes_used(), used_before);
    EXPECT_LE(rel_l2_error(re, im, in_re, in_im), 3.0 * fft2d_error_bound(rows, cols));
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(mine[i], static_cast<float>(i));
    }

    // Too little room left after the caller's allocation: throws, data and
    // arena state untouched.
    basalt::MemoryArena cramped(fft2d_scratch_bytes(rows, cols));
    ASSERT_NE(cramped.allocate_array<float>(64), nullptr);
    const size_t cramped_used = cramped.bytes_used();
    const std::vector<float> before_re = re, before_im = im;
    EXPECT_THROW(fft2d_forward(re.data(), im.data(), rows, cols, cramped), std::bad_alloc);
    EXPECT_EQ(re, before_re);
    EXPECT_EQ(im, before_im);
    EXPECT_EQ(cramped.bytes_used(), cramped_used);
}

TEST(FFT2DScratchTest, ScratchBytesHelper) {
    static_assert(fft2d_scratch_bytes(0, 64) == 0, "empty matrix needs no scratch");
    static_assert(fft2d_scratch_bytes(64, 0) == 0, "empty matrix needs no scratch");
    static_assert(fft2d_scratch_bytes(1, 1) > 2 * sizeof(float), "must cover alignment slack");
    static_assert(fft2d_scratch_bytes(std::numeric_limits<size_t>::max(), 2) ==
                      std::numeric_limits<size_t>::max(),
                  "overflow saturates");

    // The bound must be sufficient for any shape, including ones whose planes
    // are not a multiple of the 64-byte arena alignment.
    for (const Shape2D s : {Shape2D{1, 1}, Shape2D{1, 2}, Shape2D{2, 2}, Shape2D{4, 1},
                            Shape2D{8, 8}, Shape2D{1, 64}, Shape2D{64, 1}, Shape2D{128, 32}}) {
        std::vector<float> re(s.rows * s.cols, 1.0f), im(s.rows * s.cols, 0.0f);
        basalt::MemoryArena arena(fft2d_scratch_bytes(s.rows, s.cols));
        // Offset the arena start by a prior allocation of an awkward size would
        // change alignment; the helper is documented for a fresh/empty arena.
        EXPECT_NO_THROW(fft2d_forward(re.data(), im.data(), s.rows, s.cols, arena))
            << s.rows << "x" << s.cols;
    }
}

// ---------- Argument validation ----------

TEST(FFT2DValidationTest, NonPowerOfTwoDimensionsThrowAndLeaveDataUntouched) {
    const Shape2D bad[] = {{12, 16}, {16, 12}, {3, 5}, {1000, 1024}, {1024, 1000}, {6, 6}, {5, 8}};
    for (const Shape2D s : bad) {
        const size_t total = s.rows * s.cols;
        std::vector<float> re(total, 1.0f), im(total, 2.0f); // exact-size heap blocks
        basalt::MemoryArena scratch(total * 16 + 4096);
        EXPECT_THROW(fft2d_forward(re.data(), im.data(), s.rows, s.cols, scratch),
                     std::invalid_argument)
            << s.rows << "x" << s.cols;
        EXPECT_THROW(fft2d_inverse(re.data(), im.data(), s.rows, s.cols, scratch,
                                   FFT2DConfig{FFT2DSchedule::LegacyColumnFirst, 64}),
                     std::invalid_argument)
            << s.rows << "x" << s.cols;
        EXPECT_THROW(fft2d_forward(re.data(), im.data(), s.rows, s.cols, scratch,
                                   FFT2DConfig{FFT2DSchedule::FourStepRowFirst, 64}),
                     std::invalid_argument)
            << s.rows << "x" << s.cols;
        EXPECT_THROW(fft2d_forward(re.data(), im.data(), s.rows, s.cols, scratch,
                                   FFT2DConfig{FFT2DSchedule::RowFirstColumnStrips, 64}),
                     std::invalid_argument)
            << s.rows << "x" << s.cols;
        EXPECT_EQ(re, std::vector<float>(total, 1.0f));
        EXPECT_EQ(im, std::vector<float>(total, 2.0f));
        EXPECT_EQ(scratch.bytes_used(), 0u);
    }
}

TEST(FFT2DValidationTest, NullAliasedAndEmptyArguments) {
    std::vector<float> buf(64, 0.0f);
    basalt::MemoryArena scratch(fft2d_scratch_bytes(8, 8));
    EXPECT_THROW(fft2d_forward(nullptr, buf.data(), 8, 8, scratch), std::invalid_argument);
    EXPECT_THROW(fft2d_forward(buf.data(), nullptr, 8, 8, scratch), std::invalid_argument);
    EXPECT_THROW(fft2d_inverse(buf.data(), buf.data(), 8, 8, scratch), std::invalid_argument);

    // Empty matrix: documented no-op.
    EXPECT_NO_THROW(fft2d_forward(nullptr, nullptr, 0, 8, scratch));
    EXPECT_NO_THROW(fft2d_inverse(nullptr, nullptr, 8, 0, scratch));

    // Dimensions whose product overflows size_t.
    const size_t huge = size_t{1} << 62;
    EXPECT_THROW(fft2d_forward(buf.data(), buf.data() + 32, huge, huge, scratch),
                 std::invalid_argument);
}

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
            double x = static_cast<double>(c) / static_cast<double>(cols);
            double y = static_cast<double>(r) / static_cast<double>(rows);
            data.real[r * cols + c] =
                static_cast<float>(std::sin(2.0 * M_PI * 3.0 * x) * std::cos(2.0 * M_PI * 2.0 * y));
            data.imag[r * cols + c] = 0.0f;
        }
    }

    // Save original
    std::vector<float> original(data.real, data.real + total);

    // Forward then inverse; no scratch.reset() in between (fft2d releases it).
    fft2d_forward(data.real, data.imag, rows, cols, scratch);
    fft2d_inverse(data.real, data.imag, rows, cols, scratch);

    // Check reconstruction (relative L2 error)
    const std::vector<double> ref_re(original.begin(), original.end());
    const std::vector<double> ref_im(total, 0.0);
    const std::vector<float> got_re(data.real, data.real + total);
    const std::vector<float> got_im(data.imag, data.imag + total);
    EXPECT_LE(rel_l2_error(got_re, got_im, ref_re, ref_im), 2.0 * fft2d_error_bound(rows, cols))
        << rows << "x" << cols;
}

INSTANTIATE_TEST_SUITE_P(FFT2DSizes, FFT2DRoundTripTest,
                         ::testing::Values(std::make_pair(8, 8), std::make_pair(16, 16),
                                           std::make_pair(16, 32), std::make_pair(32, 16),
                                           std::make_pair(64, 64)));

// ---------- 2D Impulse response ----------

TEST(FFT2DTest, ImpulseResponse) {
    constexpr size_t rows = 8, cols = 8;
    constexpr size_t total = rows * cols;

    basalt::MemoryArena data_arena(total * 2 * sizeof(float) + 4096);
    basalt::MemoryArena scratch(total * 2 * sizeof(float) + 4096);

    ComplexSoA data(data_arena, total);
    data.clear();
    data.real[0] = 1.0f; // 2D delta

    fft2d_forward(data.real, data.imag, rows, cols, scratch);

    // 2D FFT of delta should be flat: all bins = 1.0 + 0.0i
    for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(data.real[i], 1.0f, 1e-6f) << "Real not 1.0 at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, 1e-6f) << "Imag not 0.0 at bin " << i;
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

    // DC bin (0,0) should be rows*cols = 256; tolerances scale with the DC value
    const float tol = static_cast<float>(total) * 8.0f * FLT_EPSILON;
    EXPECT_NEAR(data.real[0], static_cast<float>(total), tol);
    EXPECT_NEAR(data.imag[0], 0.0f, tol);

    // All other bins should be ~0
    for (size_t i = 1; i < total; ++i) {
        EXPECT_NEAR(data.real[i], 0.0f, tol) << "Nonzero real at bin " << i;
        EXPECT_NEAR(data.imag[i], 0.0f, tol) << "Nonzero imag at bin " << i;
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
        data.real[i] = static_cast<float>(std::sin(0.1 * static_cast<double>(i)) *
                                          std::cos(0.03 * static_cast<double>(i)));
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

    EXPECT_NEAR(freq_energy, time_energy, time_energy * 1e-4) << "2D Parseval's theorem violated";
}

TEST(FFT2DTest, ScheduleVariantsMatch) {
    constexpr size_t rows = 32;
    constexpr size_t cols = 64;
    constexpr size_t total = rows * cols;
    const size_t arena_bytes = fft2d_scratch_bytes(rows, cols);

    std::vector<float> input_real(total);
    std::vector<float> input_imag(total, 0.0f);
    for (size_t i = 0; i < total; ++i) {
        input_real[i] = static_cast<float>(std::sin(0.07 * static_cast<double>(i)) +
                                           0.25 * std::cos(0.19 * static_cast<double>(i)));
    }

    std::vector<float> legacy_real = input_real;
    std::vector<float> legacy_imag = input_imag;
    std::vector<float> four_real = input_real;
    std::vector<float> four_imag = input_imag;
    std::vector<float> strips_real = input_real;
    std::vector<float> strips_imag = input_imag;
    std::vector<float> auto_real = input_real;
    std::vector<float> auto_imag = input_imag;

    {
        basalt::MemoryArena scratch(arena_bytes);
        fft2d_forward(legacy_real.data(), legacy_imag.data(), rows, cols, scratch,
                      FFT2DConfig{FFT2DSchedule::LegacyColumnFirst, 128});
    }
    {
        basalt::MemoryArena scratch(arena_bytes);
        fft2d_forward(four_real.data(), four_imag.data(), rows, cols, scratch,
                      FFT2DConfig{FFT2DSchedule::FourStepRowFirst, 128});
    }
    {
        basalt::MemoryArena scratch(arena_bytes);
        fft2d_forward(strips_real.data(), strips_imag.data(), rows, cols, scratch,
                      FFT2DConfig{FFT2DSchedule::RowFirstColumnStrips, 128});
    }
    {
        basalt::MemoryArena scratch(arena_bytes);
        fft2d_forward(auto_real.data(), auto_imag.data(), rows, cols, scratch);
    }

    // Different schedules sum in a different order: agree to the FFT error bound.
    EXPECT_LE(rel_l2_error(four_real, four_imag, legacy_real, legacy_imag),
              fft2d_error_bound(rows, cols));
    EXPECT_LE(rel_l2_error(strips_real, strips_imag, legacy_real, legacy_imag),
              fft2d_error_bound(rows, cols));
    // Auto resolves to the column-strip schedule for this size, so it must match bit for bit.
    EXPECT_EQ(auto_real, strips_real);
    EXPECT_EQ(auto_imag, strips_imag);
}

// ---------- Runtime dispatch: every cap must give correct transforms ----------

class FFT2DSimdCapTest : public ::testing::TestWithParam<SimdMode> {};

INSTANTIATE_TEST_SUITE_P(Caps, FFT2DSimdCapTest,
                         ::testing::Values(SimdMode::Scalar, SimdMode::AVX2, SimdMode::AVX512),
                         [](const ::testing::TestParamInfo<SimdMode>& param_info) {
                             switch (param_info.param) {
                             case SimdMode::Scalar:
                                 return std::string("scalar");
                             case SimdMode::AVX2:
                                 return std::string("avx2");
                             default:
                                 return std::string("avx512");
                             }
                         });

TEST_P(FFT2DSimdCapTest, MatchesDoublePrecisionReferenceAndRoundTrips) {
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    const Shape2D shapes[] = {{1, 16},  {16, 1},   {2, 2},    {8, 8},
                              {16, 32}, {64, 128}, {128, 32}, {256, 256}};
    for (const Shape2D& shape : shapes) {
        for (FFT2DSchedule schedule :
             {FFT2DSchedule::LegacyColumnFirst, FFT2DSchedule::FourStepRowFirst,
              FFT2DSchedule::RowFirstColumnStrips, FFT2DSchedule::Auto}) {
            const FFT2DConfig config{schedule, 96};
            const size_t rows = shape.rows, cols = shape.cols, total = rows * cols;
            std::vector<float> in_re(total), in_im(total);
            fill_random(in_re, in_im, static_cast<uint32_t>(rows * 17 + cols));

            std::vector<double> ref_re, ref_im;
            reference_dft2d(in_re, in_im, rows, cols, ref_re, ref_im);

            std::vector<float> re = in_re, im = in_im;
            basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols, config));
            fft2d_forward(re.data(), im.data(), rows, cols, scratch, config);
            EXPECT_LE(rel_l2_error(re, im, ref_re, ref_im), fft2d_error_bound(rows, cols))
                << rows << "x" << cols << " " << schedule_name(schedule);

            fft2d_inverse(re.data(), im.data(), rows, cols, scratch, config);
            EXPECT_LE(rel_l2_error(re, im, in_re, in_im), 2.0 * fft2d_error_bound(rows, cols))
                << "round trip " << rows << "x" << cols << " " << schedule_name(schedule);
        }
    }
}

TEST_P(FFT2DSimdCapTest, ColumnStripsMatchReferenceOnAllStageShapes) {
    // Column lengths 1..4096 cover every stage grouping of the strip kernel
    // (fused first two stages, radix-2^2 pairs, a trailing single stage, and
    // the cache-blocked / whole-strip phases); widths 1..512 cover the scalar
    // strips (< 8 columns), one 8-lane group and multi-strip matrices.
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    const Shape2D shapes[] = {{1, 1},     {2, 1},    {4, 1},    {2, 8},     {8, 4},
                              {4, 8},     {16, 4},   {64, 2},   {32, 16},   {128, 8},
                              {256, 16},  {512, 64}, {1024, 4}, {2048, 8},  {4096, 2},
                              {256, 128}, {64, 512}, {8, 2048}, {128, 128}, {512, 32}};
    for (const Shape2D& shape : shapes) {
        const FFT2DConfig config{FFT2DSchedule::RowFirstColumnStrips, 0};
        const size_t rows = shape.rows, cols = shape.cols, total = rows * cols;
        std::vector<float> in_re(total), in_im(total);
        fill_random(in_re, in_im, static_cast<uint32_t>(rows * 31 + cols));

        std::vector<double> ref_re, ref_im;
        reference_dft2d(in_re, in_im, rows, cols, ref_re, ref_im);

        std::vector<float> re = in_re, im = in_im;
        basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols, config));
        fft2d_forward(re.data(), im.data(), rows, cols, scratch, config);
        EXPECT_LE(rel_l2_error(re, im, ref_re, ref_im), fft2d_error_bound(rows, cols))
            << "forward " << rows << "x" << cols;
        EXPECT_EQ(scratch.bytes_used(), 0u);

        fft2d_inverse(re.data(), im.data(), rows, cols, scratch, config);
        EXPECT_LE(rel_l2_error(re, im, in_re, in_im), 2.0 * fft2d_error_bound(rows, cols))
            << "round trip " << rows << "x" << cols;

        // Inverse of the reference spectrum must give back the input (checks the
        // conjugated twiddles and the 1/(rows*cols) scale independently).
        std::vector<float> spec_re(ref_re.begin(), ref_re.end());
        std::vector<float> spec_im(ref_im.begin(), ref_im.end());
        fft2d_inverse(spec_re.data(), spec_im.data(), rows, cols, scratch, config);
        EXPECT_LE(rel_l2_error(spec_re, spec_im, in_re, in_im), 2.0 * fft2d_error_bound(rows, cols))
            << "inverse " << rows << "x" << cols;
    }
}

TEST_P(FFT2DSimdCapTest, PlaneWavePeakLandsInExpectedBin) {
    basalt::kernel::detail::ScopedSimdCap cap(GetParam());
    const size_t rows = 32, cols = 64, k0 = 3, l0 = 5;
    std::vector<float> re(rows * cols), im(rows * cols);
    fill_plane_wave(re, im, rows, cols, k0, l0);
    basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));
    fft2d_forward(re.data(), im.data(), rows, cols, scratch);
    EXPECT_NEAR(re[k0 * cols + l0], static_cast<float>(rows * cols), 1e-2f);
    EXPECT_NEAR(im[k0 * cols + l0], 0.0f, 1e-2f);
    EXPECT_NEAR(re[0], 0.0f, 1e-2f);
}

TEST(FFT2DDispatchTest, ScalarAndWidePathsAgree) {
    const size_t rows = 128, cols = 256, total = rows * cols;
    std::vector<float> in_re(total), in_im(total);
    fill_random(in_re, in_im, 7);

    std::vector<float> s_re = in_re, s_im = in_im, w_re = in_re, w_im = in_im;
    {
        basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));
        basalt::kernel::detail::ScopedSimdCap cap(SimdMode::Scalar);
        fft2d_forward(s_re.data(), s_im.data(), rows, cols, scratch);
    }
    {
        basalt::MemoryArena scratch(fft2d_scratch_bytes(rows, cols));
        fft2d_forward(w_re.data(), w_im.data(), rows, cols, scratch); // best available path
    }
    EXPECT_LE(rel_l2_error(s_re, s_im, w_re, w_im), fft2d_error_bound(rows, cols));
}
