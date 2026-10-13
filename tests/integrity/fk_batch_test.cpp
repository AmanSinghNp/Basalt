// Tests for the batch API fk_filter_gathers (basalt/fk.hpp) and its worker pool.
// Contract: docs/design/FK_PIPELINE.md, section 4, and the fk_filter_gathers comments in fk.hpp.

#include "basalt/fk.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#if defined(__SSE__)
#include "basalt/kernel/cpu_state.hpp"
#include <xmmintrin.h>
#endif

using basalt::FKMode;
using basalt::FKMuteParams;
using basalt::fk_filter_gather;
using basalt::fk_filter_gathers;

namespace {

constexpr float kSentinel = -12345.0f;

FKMuteParams batch_params() {
    FKMuteParams p;
    p.dt = 0.002;
    p.dx = 5.0;
    p.v_min = 500.0;
    p.v_max = 2000.0;
    p.taper = 0.1;
    p.mode = FKMode::Reject;
    p.threads = 1;
    return p;
}

// `count` normally distributed samples, seeded.
std::vector<float> random_values(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(count);
    for (float& x : v) {
        x = dist(rng);
    }
    return v;
}

// Bitwise equality. memcmp rather than operator==, so that -0.0 and +0.0 and NaN payloads count.
bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size()
        && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// The reference: fk_filter_gather with threads = 1 on each gather in turn.
std::vector<float> per_gather_reference(const std::vector<float>& in, size_t ngathers,
                                        size_t ntr, size_t ns, FKMuteParams params) {
    params.threads = 1;
    std::vector<float> out(in.size(), kSentinel);
    const size_t len = ntr * ns;
    for (size_t g = 0; g < ngathers; ++g) {
        fk_filter_gather(in.data() + g * len, out.data() + g * len, ntr, ns, params);
    }
    return out;
}

// Runs fk_filter_gathers out of place into a sentinel-filled buffer.
std::vector<float> batch_out_of_place(const std::vector<float>& in, size_t ngathers, size_t ntr,
                                      size_t ns, const FKMuteParams& params) {
    std::vector<float> out(in.size(), kSentinel);
    fk_filter_gathers(in.data(), out.data(), ngathers, ntr, ns, params);
    return out;
}

// Runs fk_filter_gathers in place on a copy of `in`.
std::vector<float> batch_in_place(const std::vector<float>& in, size_t ngathers, size_t ntr,
                                  size_t ns, const FKMuteParams& params) {
    std::vector<float> buf = in;
    fk_filter_gathers(buf.data(), buf.data(), ngathers, ntr, ns, params);
    return buf;
}

#if defined(__linux__)
// User plus system CPU seconds used by this process, from /proc/self/stat.
double process_cpu_seconds() {
    std::ifstream stat("/proc/self/stat");
    std::string line;
    std::getline(stat, line);
    // Fields after the parenthesis that closes the command name. The state is field 3, so
    // utime (field 14) and stime (field 15) are tokens 11 and 12 of what follows.
    const size_t close = line.rfind(')');
    std::istringstream rest(line.substr(close + 2));
    std::vector<std::string> tok;
    for (std::string t; rest >> t;) {
        tok.push_back(t);
    }
    const double ticks = std::stod(tok.at(11)) + std::stod(tok.at(12));
    return ticks / static_cast<double>(sysconf(_SC_CLK_TCK));
}

// Resident set size of this process in kiB, from /proc/self/status.
long resident_kib() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::stol(line.substr(6));
        }
    }
    return -1;
}
#endif

struct Shape {
    size_t ntr;
    size_t ns;
};

} // namespace

// --- Bitwise equality with the per-gather reference -----------------------------------------

TEST(FKBatch, MatchesPerGatherSingleThreadBitwise) {
    const std::vector<Shape> shapes = {{1, 5}, {7, 13}, {33, 77}, {64, 129}, {240, 301}};
    for (const Shape& s : shapes) {
        for (const size_t ngathers : {size_t{2}, size_t{5}}) {
            for (const FKMode mode : {FKMode::Reject, FKMode::Pass}) {
                FKMuteParams params = batch_params();
                params.mode = mode;
                const std::vector<float> in = random_values(ngathers * s.ntr * s.ns, 11);
                const std::vector<float> expected =
                    per_gather_reference(in, ngathers, s.ntr, s.ns, params);
                for (const size_t threads : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
                    params.threads = threads;
                    EXPECT_TRUE(bit_equal(batch_out_of_place(in, ngathers, s.ntr, s.ns, params),
                                          expected))
                        << "shape " << s.ntr << "x" << s.ns << " gathers " << ngathers
                        << " threads " << threads << " mode " << static_cast<int>(mode);
                }
            }
        }
    }

    // Above the parallel threshold of the weight table build (ntr_pad * nfreq >= 262144 in
    // fk_pipeline.cpp), so with threads > 1 the shared table is built on several threads.
    // 1001 x 1500 pads to about 1.7 M weights.
    {
        const Shape s{1001, 1500};
        const size_t ngathers = 3;
        for (const FKMode mode : {FKMode::Reject, FKMode::Pass}) {
            FKMuteParams params = batch_params();
            params.mode = mode;
            const std::vector<float> in = random_values(ngathers * s.ntr * s.ns, 13);
            const std::vector<float> expected =
                per_gather_reference(in, ngathers, s.ntr, s.ns, params);
            for (const size_t threads : {size_t{1}, size_t{4}}) {
                params.threads = threads;
                EXPECT_TRUE(bit_equal(batch_out_of_place(in, ngathers, s.ntr, s.ns, params),
                                      expected))
                    << "large shape threads " << threads << " mode " << static_cast<int>(mode);
            }
        }
    }
}

TEST(FKBatch, InPlaceMatchesPerGatherBitwise) {
    const std::vector<Shape> shapes = {{7, 13}, {33, 77}, {240, 301}};
    for (const Shape& s : shapes) {
        const size_t ngathers = 6;
        FKMuteParams params = batch_params();
        const std::vector<float> in = random_values(ngathers * s.ntr * s.ns, 23);
        const std::vector<float> expected = per_gather_reference(in, ngathers, s.ntr, s.ns, params);
        for (const size_t threads : {size_t{1}, size_t{3}, size_t{4}}) {
            params.threads = threads;
            EXPECT_TRUE(bit_equal(batch_in_place(in, ngathers, s.ntr, s.ns, params), expected))
                << "shape " << s.ntr << "x" << s.ns << " threads " << threads;
        }
    }
}

TEST(FKBatch, ColumnClassCasesMatchPerGatherBitwise) {
    // The identity configuration makes every column "ones", and the Pass mode with an empty band
    // makes every column "zeros". The default band mixes the classes.
    const size_t ntr = 40;
    const size_t ns = 96;
    const size_t ngathers = 3;
    const std::vector<float> in = random_values(ngathers * ntr * ns, 31);

    FKMuteParams identity = batch_params();
    identity.v_min = 1e29;
    identity.v_max = 1e30;
    identity.taper = 0.0;
    identity.edge_taper_traces = 0;

    FKMuteParams all_zero = identity;
    all_zero.mode = FKMode::Pass;

    FKMuteParams mixed = batch_params();
    mixed.pad_factor = 2.0;
    mixed.edge_taper_samples = 3;

    for (const FKMuteParams& params : {identity, all_zero, mixed}) {
        const std::vector<float> expected = per_gather_reference(in, ngathers, ntr, ns, params);
        for (const size_t threads : {size_t{1}, size_t{4}}) {
            FKMuteParams p = params;
            p.threads = threads;
            EXPECT_TRUE(bit_equal(batch_out_of_place(in, ngathers, ntr, ns, p), expected))
                << "threads " << threads;
        }
    }
}

TEST(FKBatch, ZeroThreadsUsesHardwareConcurrencyAndMatches) {
    const size_t ngathers = 4;
    const size_t ntr = 50;
    const size_t ns = 150;
    const std::vector<float> in = random_values(ngathers * ntr * ns, 41);
    FKMuteParams params = batch_params();
    const std::vector<float> expected = per_gather_reference(in, ngathers, ntr, ns, params);
    params.threads = 0;
    EXPECT_TRUE(bit_equal(batch_out_of_place(in, ngathers, ntr, ns, params), expected));
}

TEST(FKBatch, SingleGatherIsExactlyFilterGather) {
    const size_t ntr = 60;
    const size_t ns = 200;
    const std::vector<float> in = random_values(ntr * ns, 51);
    for (const size_t threads : {size_t{1}, size_t{4}}) {
        FKMuteParams params = batch_params();
        params.threads = threads;
        const std::vector<float> expected = fk_filter_gather(in, ntr, ns, params);
        EXPECT_TRUE(bit_equal(batch_out_of_place(in, 1, ntr, ns, params), expected))
            << "threads " << threads;
        EXPECT_TRUE(bit_equal(batch_in_place(in, 1, ntr, ns, params), expected))
            << "threads " << threads;
    }
}

// --- Non-finite input ------------------------------------------------------------------------

TEST(FKBatch, NonFiniteInLastGatherThrowsAndLeavesAliasedBufferUntouched) {
    const size_t ngathers = 5;
    const size_t ntr = 20;
    const size_t ns = 60;
    const size_t len = ntr * ns;
    FKMuteParams params = batch_params();
    params.threads = 4;

    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity()}) {
        std::vector<float> buf = random_values(ngathers * len, 61);
        buf[(ngathers - 1) * len + len / 2] = bad; // last gather, middle sample
        const std::vector<float> before = buf;

        // Bitwise snapshot, so that NaN payloads are compared as well.
        try {
            fk_filter_gathers(buf.data(), buf.data(), ngathers, ntr, ns, params);
            ADD_FAILURE() << "no exception for a non-finite sample";
        } catch (const std::invalid_argument& e) {
            EXPECT_EQ(std::string(e.what()), "fk: input contains NaN or infinite samples");
        }
        EXPECT_TRUE(bit_equal(buf, before)) << "aliased input was modified";
    }
}

TEST(FKBatch, NonFiniteInFirstGatherLeavesOutputUntouched) {
    const size_t ngathers = 4;
    const size_t ntr = 16;
    const size_t ns = 48;
    const size_t len = ntr * ns;
    FKMuteParams params = batch_params();
    params.threads = 3;
    std::vector<float> in = random_values(ngathers * len, 71);
    in[0] = std::numeric_limits<float>::infinity();
    std::vector<float> out(ngathers * len, kSentinel);
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), ngathers, ntr, ns, params),
                 std::invalid_argument);
    EXPECT_TRUE(bit_equal(out, std::vector<float>(ngathers * len, kSentinel)));
}

// At one thread the batch skips the per-gather finiteness test, so the scan alone must catch a
// non-finite sample. Checks a middle gather, out of place and in place.
TEST(FKBatch, NonFiniteInMiddleGatherIsCaughtAtOneThread) {
    const size_t ngathers = 4;
    const size_t ntr = 18;
    const size_t ns = 52;
    const size_t len = ntr * ns;
    FKMuteParams params = batch_params();
    params.threads = 1;
    std::vector<float> in = random_values(ngathers * len, 91);
    in[2 * len + 7] = std::numeric_limits<float>::quiet_NaN();

    std::vector<float> out(ngathers * len, kSentinel);
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), ngathers, ntr, ns, params),
                 std::invalid_argument);
    EXPECT_TRUE(bit_equal(out, std::vector<float>(ngathers * len, kSentinel)));

    const std::vector<float> before = in;
    EXPECT_THROW(fk_filter_gathers(in.data(), in.data(), ngathers, ntr, ns, params),
                 std::invalid_argument);
    EXPECT_TRUE(bit_equal(in, before));
}

// --- Argument handling -----------------------------------------------------------------------

TEST(FKBatch, ZeroGathersIsANoOpAfterValidation) {
    const size_t ntr = 12;
    const size_t ns = 30;
    std::vector<float> in = random_values(ntr * ns, 81);
    std::vector<float> out(ntr * ns, kSentinel);
    FKMuteParams params = batch_params();
    params.threads = 4;
    EXPECT_NO_THROW(fk_filter_gathers(in.data(), out.data(), 0, ntr, ns, params));
    EXPECT_TRUE(bit_equal(out, std::vector<float>(ntr * ns, kSentinel)));

    // Validation still applies when there are no gathers to filter.
    FKMuteParams bad = params;
    bad.taper = 1.0;
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), 0, ntr, ns, bad), std::invalid_argument);
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), 0, 0, ns, params), std::invalid_argument);
    EXPECT_THROW(fk_filter_gathers(nullptr, out.data(), 0, ntr, ns, params), std::invalid_argument);
}

TEST(FKBatch, InvalidArgumentsThrowInvalidArgument) {
    const size_t ntr = 8;
    const size_t ns = 16;
    std::vector<float> in = random_values(4 * ntr * ns, 91);
    std::vector<float> out(in.size());
    const FKMuteParams good = batch_params();

    FKMuteParams p = good;
    p.dt = 0.0;
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), 4, ntr, ns, p), std::invalid_argument);
    p = good;
    p.taper = 1.0;
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), 4, ntr, ns, p), std::invalid_argument);
    p = good;
    p.pad_factor = 0.5;
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), 4, ntr, ns, p), std::invalid_argument);
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), 4, 0, ns, good), std::invalid_argument);
    EXPECT_THROW(fk_filter_gathers(in.data(), out.data(), 4, ntr, 0, good), std::invalid_argument);
    EXPECT_THROW(fk_filter_gathers(nullptr, out.data(), 4, ntr, ns, good), std::invalid_argument);
    EXPECT_THROW(fk_filter_gathers(in.data(), nullptr, 4, ntr, ns, good), std::invalid_argument);
}

TEST(FKBatch, ErrorPrecedenceMatchesFilterGather) {
    // When several arguments are wrong, fk_filter_gathers reports the same error as
    // fk_filter_gather, whatever the gather count.
    const size_t ntr = 8;
    const size_t ns = 16;
    std::vector<float> buf(4 * ntr * ns, 1.0f);
    FKMuteParams tiny_dt = batch_params();
    tiny_dt.dt = 1e-310; // the frequency step overflows: rejected by the geometry checks

    const auto message = [](const std::function<void()>& call) {
        try {
            call();
        } catch (const std::invalid_argument& e) {
            return std::string(e.what());
        }
        return std::string("no exception");
    };
    // Null pointer and a bad geometry: the pointer is reported first.
    EXPECT_EQ(message([&] { fk_filter_gather(nullptr, buf.data(), ntr, ns, tiny_dt); }),
              message([&] { fk_filter_gathers(nullptr, buf.data(), 4, ntr, ns, tiny_dt); }));
    EXPECT_EQ(message([&] { fk_filter_gather(nullptr, buf.data(), ntr, ns, tiny_dt); }),
              "fk: null input or output pointer");
    // Zero traces and a bad geometry: the dimensions are reported first.
    EXPECT_EQ(message([&] { fk_filter_gather(buf.data(), buf.data(), 0, ns, tiny_dt); }),
              message([&] { fk_filter_gathers(buf.data(), buf.data(), 4, 0, ns, tiny_dt); }));
    // Bad geometry with valid pointers: the same geometry message.
    EXPECT_EQ(message([&] { fk_filter_gather(buf.data(), buf.data(), ntr, ns, tiny_dt); }),
              message([&] { fk_filter_gathers(buf.data(), buf.data(), 4, ntr, ns, tiny_dt); }));
    EXPECT_NE(message([&] { fk_filter_gathers(buf.data(), buf.data(), 4, ntr, ns, tiny_dt); }),
              "no exception");
}

TEST(FKBatch, OverflowOfTheBatchSizeIsRejected) {
    // ngathers * ntraces * nsamples does not fit in size_t. The check runs before any work, so
    // the buffers can be small.
    float dummy = 0.0f;
    const size_t ntr = 1000;
    const size_t ns = 1000;
    const size_t too_many = std::numeric_limits<size_t>::max() / (ntr * ns) + 1;
    EXPECT_THROW(fk_filter_gathers(&dummy, &dummy, too_many, ntr, ns, batch_params()),
                 std::invalid_argument);
    // ntraces * nsamples itself overflows, so the padded grid is too large (rejected by the
    // geometry checks, which run first).
    const size_t big = std::numeric_limits<size_t>::max() / 2 + 1;
    EXPECT_THROW(fk_filter_gathers(&dummy, &dummy, 1, big, 3, batch_params()),
                 std::invalid_argument);
}

// --- Concurrency -----------------------------------------------------------------------------

TEST(FKBatch, ConcurrentCallsFromTwoThreadsAreSafe) {
    const size_t ngathers = 6;
    const size_t ntr = 40;
    const size_t ns = 120;
    const std::vector<float> in_a = random_values(ngathers * ntr * ns, 101);
    const std::vector<float> in_b = random_values(ngathers * ntr * ns, 102);
    FKMuteParams params = batch_params();
    const std::vector<float> expected_a = per_gather_reference(in_a, ngathers, ntr, ns, params);
    const std::vector<float> expected_b = per_gather_reference(in_b, ngathers, ntr, ns, params);

    params.threads = 3;
    bool ok_a = true;
    bool ok_b = true;
    const auto worker = [&](const std::vector<float>& in, const std::vector<float>& expected,
                            bool& ok) {
        for (int rep = 0; rep < 8; ++rep) {
            ok = ok && bit_equal(batch_out_of_place(in, ngathers, ntr, ns, params), expected);
        }
    };
    std::thread a(worker, std::cref(in_a), std::cref(expected_a), std::ref(ok_a));
    std::thread b(worker, std::cref(in_b), std::cref(expected_b), std::ref(ok_b));
    a.join();
    b.join();
    EXPECT_TRUE(ok_a);
    EXPECT_TRUE(ok_b);
}

// --- Worker pool: idle behaviour -------------------------------------------------------------

#if defined(__linux__)
TEST(FKBatchPool, IdleWorkersBlockAndExitWithoutSpinning) {
    const size_t ngathers = 16;
    const size_t ntr = 240;
    const size_t ns = 1001;
    FKMuteParams params = batch_params();
    params.threads = 4;
    const std::vector<float> in = random_values(ngathers * ntr * ns, 111);
    std::vector<float> buf = in;
    fk_filter_gathers(buf.data(), buf.data(), ngathers, ntr, ns, params);

    // Process CPU time over the 3 s after the call. The workers wait on a condition variable
    // (and exit after 2 s without work), so the window costs almost nothing. A pool that polls
    // used about 0.6 CPU-s per second per worker.
    const double cpu0 = process_cpu_seconds();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    const double idle_cpu = process_cpu_seconds() - cpu0;
    // The spec bar is under 2 % of one core over the window (0.06 s of 3 s).
    EXPECT_LT(idle_cpu, 0.06) << "CPU over 3 s of idleness: " << idle_cpu << " s";
}

TEST(FKBatchPool, IdleWorkersReleaseWorkspaceAndAreRecreatedOnDemand) {
    const size_t ngathers = 12;
    const size_t ntr = 1001;
    const size_t ns = 1500;
    FKMuteParams params = batch_params();
    params.threads = 4;
    params.dt = 0.004;
    params.dx = 25.0;
    params.v_min = -1500.0;
    params.v_max = 1500.0;
    const std::vector<float> in = random_values(ngathers * ntr * ns, 121);
    const std::vector<float> expected = per_gather_reference(in, ngathers, ntr, ns, params);

    const long before = resident_kib();
    const std::vector<float> first = batch_out_of_place(in, ngathers, ntr, ns, params);
    const long busy = resident_kib();
    EXPECT_TRUE(bit_equal(first, expected));

    // Idle past the timeout. Every pool worker's workspace (about 27 MB each at this size) must be
    // returned, so the resident set falls back by more than one worker's share.
    std::this_thread::sleep_for(std::chrono::milliseconds(3500));
    const long idle = resident_kib();
    EXPECT_LT(idle, busy - 10L * 1024) << "RSS busy " << busy << " kiB, idle " << idle
                                       << " kiB, before " << before << " kiB";

    // The next call starts workers again and gives the same result.
    EXPECT_TRUE(bit_equal(batch_out_of_place(in, ngathers, ntr, ns, params), expected));
}

TEST(FKBatchPool, BackToBackCallsGiveIdenticalResults) {
    // Calls with no idle gap between them (the workers are reused while they are still alive)
    // all give the same result.
    const size_t ngathers = 8;
    const size_t ntr = 64;
    const size_t ns = 300;
    FKMuteParams params = batch_params();
    params.threads = 4;
    const std::vector<float> in = random_values(ngathers * ntr * ns, 131);
    const std::vector<float> expected = per_gather_reference(in, ngathers, ntr, ns, params);
    for (int rep = 0; rep < 20; ++rep) {
        ASSERT_TRUE(bit_equal(batch_out_of_place(in, ngathers, ntr, ns, params), expected))
            << "call " << rep;
    }
}
#endif

#if defined(__SSE__)
// A pool worker runs under the floating-point mode of the thread that created it. The batch must
// still give the caller's bits. Here the workers are created by a thread with FTZ and DAZ cleared,
// and the batch then runs on a thread with them set. Inputs are at subnormal-range scale, so a mode
// mismatch changes output bits.
TEST(FKBatchPool, WorkersUseTheCallersFloatingPointMode) {
    const size_t ngathers = 6;
    const size_t ntr = 40;
    const size_t ns = 120;
    const size_t len = ntr * ns;
    FKMuteParams params = batch_params();
    std::vector<float> in = random_values(ngathers * len, 141);
    for (float& x : in) {
        x *= 1e-35f;
    }

    // Workers left by earlier tests exit after kIdleTimeout (2 s) without work. Waiting that long
    // makes the workers below the ones this test creates.
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));

    params.threads = 4;
    std::thread creator([&] {
        // A new thread inherits the mode of the thread that starts it, so clear FTZ and DAZ here.
        _mm_setcsr(_mm_getcsr() & ~0x8040u);
        std::vector<float> scratch(in.size());
        fk_filter_gathers(in.data(), scratch.data(), ngathers, ntr, ns, params);
    });
    creator.join();

    // Restores the caller's mode when the test ends, even on failure.
    struct ModeRestore {
        unsigned value;
        ~ModeRestore() {
            _mm_setcsr(value);
        }
    } restore{_mm_getcsr()};
    basalt::kernel::set_flush_to_zero();
    ASSERT_TRUE(basalt::kernel::check_ftz_daz());

    FKMuteParams reference = params;
    reference.threads = 1;
    const std::vector<float> expected = per_gather_reference(in, ngathers, ntr, ns, reference);
    std::vector<float> out(in.size(), kSentinel);
    fk_filter_gathers(in.data(), out.data(), ngathers, ntr, ns, params);
    EXPECT_TRUE(bit_equal(out, expected));
}
#endif
