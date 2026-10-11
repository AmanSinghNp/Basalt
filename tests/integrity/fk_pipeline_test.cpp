// Tests for the end-to-end f-k mute pipeline (basalt/fk.hpp).
// Numerical behaviour follows docs/design/FK_PIPELINE.md, sections 2-4.

#include "basalt/fk.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using basalt::FKMode;
using basalt::FKMuteParams;
using basalt::fk_filter_gather;
using basalt::fk_mute_weight;
using basalt::fk_padded_samples;
using basalt::fk_padded_traces;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

// Band parameters tuned for the synthetic events: dx = 5 m, dt = 2 ms, 25 Hz Ricker.
FKMuteParams event_params() {
    FKMuteParams p;
    p.dt = 0.002;
    p.dx = 5.0;
    p.v_min = 500.0;
    p.v_max = 2000.0;
    p.taper = 0.1;
    p.mode = FKMode::Reject;
    return p;
}

std::vector<float> random_gather(size_t ntr, size_t ns, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(ntr * ns);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

// Subtracts the across-trace mean at every sample, so the k = 0 row vanishes.
void remove_trace_mean(std::vector<float>& d, size_t ntr, size_t ns) {
    for (size_t s = 0; s < ns; ++s) {
        double m = 0.0;
        for (size_t t = 0; t < ntr; ++t) {
            m += d[t * ns + s];
        }
        m /= static_cast<double>(ntr);
        for (size_t t = 0; t < ntr; ++t) {
            d[t * ns + s] = static_cast<float>(d[t * ns + s] - m);
        }
    }
}

// Relative RMS difference: ||a - b|| / ||b||.
double rel_rms(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(num / den);
}

// Rows [t0, t1) of a trace-major gather.
std::vector<float> trace_rows(const std::vector<float>& d, size_t ns, size_t t0, size_t t1) {
    return std::vector<float>(d.begin() + static_cast<std::ptrdiff_t>(t0 * ns),
                              d.begin() + static_cast<std::ptrdiff_t>(t1 * ns));
}

double ricker(double tau, double f0) {
    const double a = kPi * f0 * tau;
    return (1.0 - 2.0 * a * a) * std::exp(-a * a);
}

// Adds a linear event whose arrival is t0 + x / v (so v > 0 means time increases with x).
// With `hann` the amplitude follows a Hann envelope along x, which suppresses the spectral
// sidelobes that a hard truncation of the gather would spread into every velocity band.
void add_linear_event(std::vector<float>& d, size_t ntr, size_t ns, double dt, double dx,
                      double v, double t0, double amp, double f0, bool hann = false) {
    for (size_t t = 0; t < ntr; ++t) {
        const double x = static_cast<double>(t) * dx;
        const double arrival = t0 + x / v;
        const double env = hann
            ? std::pow(std::sin(kPi * (static_cast<double>(t) + 0.5) / static_cast<double>(ntr)),
                       2.0)
            : 1.0;
        for (size_t s = 0; s < ns; ++s) {
            const double tau = static_cast<double>(s) * dt - arrival;
            d[t * ns + s] += static_cast<float>(amp * env * ricker(tau, f0));
        }
    }
}

struct Event {
    double v;
    double t0;
};

// Two well-separated events: A (+1000 m/s) arrives at 0.30 s + x/1000 and
// B (-1000 m/s) at 2.00 s - x/1000. Their arrivals inside the central 50% of
// traces are at least 0.21 s apart (B - A = 1.70 - 2x/1000 s, minimum at trace 149), so the
// +-0.06 s trajectory windows never overlap.
constexpr size_t kEvNtr = 200;
constexpr size_t kEvNs = 1100;
constexpr double kEvDt = 0.002;
constexpr double kEvDx = 5.0;
constexpr double kEvF0 = 25.0;
const Event kEventA{1000.0, 0.30};
const Event kEventB{-1000.0, 2.00};

std::vector<float> two_event_gather(bool with_a, bool with_b, bool hann = false) {
    std::vector<float> d(kEvNtr * kEvNs, 0.0f);
    if (with_a) {
        add_linear_event(d, kEvNtr, kEvNs, kEvDt, kEvDx, kEventA.v, kEventA.t0, 1.0, kEvF0, hann);
    }
    if (with_b) {
        add_linear_event(d, kEvNtr, kEvNs, kEvDt, kEvDx, kEventB.v, kEventB.t0, 1.0, kEvF0, hann);
    }
    return d;
}

// Energy of `d` within +-0.06 s of the trajectory of `ev`, over the central 50% of traces.
double trajectory_energy(const std::vector<float>& d, const Event& ev) {
    const double half_win = 0.06;
    double e = 0.0;
    for (size_t t = kEvNtr / 4; t < 3 * kEvNtr / 4; ++t) {
        const double x = static_cast<double>(t) * kEvDx;
        const double arrival = ev.t0 + x / ev.v;
        for (size_t s = 0; s < kEvNs; ++s) {
            const double tau = static_cast<double>(s) * kEvDt - arrival;
            if (std::fabs(tau) <= half_win) {
                const double a = d[t * kEvNs + s];
                e += a * a;
            }
        }
    }
    return e;
}

// Energy retained along an event's trajectory, in dB (0 dB = unchanged).
double retained_db(const std::vector<float>& out, const std::vector<float>& in, const Event& ev) {
    return 10.0 * std::log10(trajectory_energy(out, ev) / trajectory_energy(in, ev));
}

// Taper weight used by the pipeline for the i-th trace/sample from an edge.
double taper_weight(size_t i, size_t n) {
    return 0.5 - 0.5 * std::cos(kPi * static_cast<double>(i + 1) / static_cast<double>(n + 1));
}

// Raised-cosine transition used by the spec, evaluated independently in the test.
double expected_rise(double v, double e, double h) {
    if (v <= e - h) {
        return 0.0;
    }
    if (v >= e + h) {
        return 1.0;
    }
    return 0.5 - 0.5 * std::cos(kPi * (v - (e - h)) / (2.0 * h));
}

bool only_prime_factors(size_t n, std::initializer_list<size_t> primes) {
    for (size_t p : primes) {
        while (n > 1 && n % p == 0) {
            n /= p;
        }
    }
    return n == 1;
}

// Direct O(n^2) evaluation of spec section 4 (edge tapers, real 2D DFT over the padded grid,
// mute weights, inverse, crop). It shares no code with pocketfft, so it checks padding offsets,
// the frequency grid and the Nyquist rows at sizes where the FFT path is otherwise only checked
// for finiteness. The inverse takes the real part of the full sum with the half spectrum
// extended by conj(Y(-i, n1 - j)), which is what irfft2 computes. Only for small gathers.
std::vector<float> direct_fk_reference(const std::vector<float>& in, size_t ntr, size_t ns,
                                       const FKMuteParams& p) {
    using cd = std::complex<double>;
    const size_t n0 = fk_padded_traces(ntr, p);
    const size_t n1 = fk_padded_samples(ns, p);
    const size_t nf = n1 / 2 + 1;

    // Unit-modulus root exp(sign * 2 pi i a / n).
    auto root = [](size_t a, size_t n, double sign) {
        const double ang = sign * 2.0 * kPi * static_cast<double>(a) / static_cast<double>(n);
        return cd(std::cos(ang), std::sin(ang));
    };

    // Step 3: tapered input on the top-left of the padded grid (zero elsewhere).
    std::vector<double> pad(n0 * n1, 0.0);
    for (size_t t = 0; t < ntr; ++t) {
        for (size_t s = 0; s < ns; ++s) {
            double w = 1.0;
            if (t < p.edge_taper_traces) {
                w *= taper_weight(t, p.edge_taper_traces);
            }
            if (ntr - 1 - t < p.edge_taper_traces) {
                w *= taper_weight(ntr - 1 - t, p.edge_taper_traces);
            }
            if (s < p.edge_taper_samples) {
                w *= taper_weight(s, p.edge_taper_samples);
            }
            if (ns - 1 - s < p.edge_taper_samples) {
                w *= taper_weight(ns - 1 - s, p.edge_taper_samples);
            }
            pad[t * n1 + s] = static_cast<double>(in[t * ns + s]) * w;
        }
    }

    // Steps 4-5: half spectrum of the padded grid, then the mute weight of each bin.
    const double df = 1.0 / (static_cast<double>(n1) * p.dt);
    const double dk = 1.0 / (static_cast<double>(n0) * p.dx);
    std::vector<cd> y(n0 * nf);
    for (size_t i = 0; i < n0; ++i) {
        const double k_index = (i <= (n0 - 1) / 2)
            ? static_cast<double>(i)
            : static_cast<double>(i) - static_cast<double>(n0);
        for (size_t j = 0; j < nf; ++j) {
            cd acc(0.0, 0.0);
            for (size_t t = 0; t < ntr; ++t) {
                for (size_t s = 0; s < ns; ++s) {
                    acc += pad[t * n1 + s] * root((i * t) % n0, n0, -1.0) *
                           root((j * s) % n1, n1, -1.0);
                }
            }
            const double w = fk_mute_weight(static_cast<double>(j) * df, k_index * dk, p);
            y[i * nf + j] = acc * w;
        }
    }

    // Step 6: full inverse sum over the grid, using the Hermitian extension of the half spectrum.
    auto yfull = [&](size_t i, size_t j) -> cd {
        if (j < nf) {
            return y[i * nf + j];
        }
        return std::conj(y[((n0 - i) % n0) * nf + (n1 - j)]);
    };
    std::vector<float> out(ntr * ns);
    const double scale = 1.0 / (static_cast<double>(n0) * static_cast<double>(n1));
    for (size_t t = 0; t < ntr; ++t) {
        for (size_t s = 0; s < ns; ++s) {
            cd acc(0.0, 0.0);
            for (size_t i = 0; i < n0; ++i) {
                for (size_t j = 0; j < n1; ++j) {
                    acc += yfull(i, j) * root((i * t) % n0, n0, 1.0) * root((j * s) % n1, n1, 1.0);
                }
            }
            out[t * ns + s] = static_cast<float>(acc.real() * scale);
        }
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Mute weight (section 3)
// ---------------------------------------------------------------------------

TEST(FKWeight, BandCentreRejectAndPass) {
    FKMuteParams p;  // [-1500, 1500], taper 0.1
    // v = 0 at the centre of the band: f = 0, k != 0.
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.02, p), 0.0);
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.02, p), 1.0);

    // Centre of [500, 2000] is v = 1250: f = -625, k = 0.5 gives v = 1250 exactly.
    FKMuteParams q = event_params();
    EXPECT_DOUBLE_EQ(fk_mute_weight(-625.0, 0.5, q), 0.0);  // Reject mutes the centre
    q.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-625.0, 0.5, q), 1.0);  // Pass keeps the centre
}

TEST(FKWeight, BandEdgesAreHalfWithTaper) {
    // taper > 0: each edge of the band has weight 0.5 (inside = rise(e) * fall(other) = 0.5).
    FKMuteParams p = event_params();  // [500, 2000], taper 0.1
    // v = 500: f = -250, k = 0.5 gives v = 500 exactly.
    EXPECT_DOUBLE_EQ(fk_mute_weight(-250.0, 0.5, p), 1.0 - 0.5);  // Reject
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-250.0, 0.5, p), 0.5);
    // v = 2000: f = -1000, k = 0.5.
    p.mode = FKMode::Reject;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-1000.0, 0.5, p), 0.5);
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-1000.0, 0.5, p), 0.5);
}

TEST(FKWeight, TransitionMatchesRaisedCosine) {
    FKMuteParams p = event_params();  // [500, 2000], taper 0.1
    p.mode = FKMode::Pass;
    // Lower edge: h = 0.5 * 0.1 * 500 = 25, transition [475, 525].
    const double h_lo = 25.0;
    const double h_hi = 100.0;  // 0.5 * 0.1 * 2000
    for (double v : {476.0, 490.0, 500.0, 510.0, 524.0}) {
        const double expect = expected_rise(v, 500.0, h_lo) *
                              (1.0 - expected_rise(v, 2000.0, h_hi));
        // v = -f / k with k = 0.5 and f = -v / 2.
        EXPECT_NEAR(fk_mute_weight(-v * 0.5, 0.5, p), expect, 1e-12) << "v = " << v;
    }
    // Upper edge: transition [1900, 2100].
    for (double v : {1910.0, 2000.0, 2050.0, 2090.0}) {
        const double expect = expected_rise(v, 500.0, h_lo) *
                              (1.0 - expected_rise(v, 2000.0, h_hi));
        EXPECT_NEAR(fk_mute_weight(-v * 0.5, 0.5, p), expect, 1e-12) << "v = " << v;
    }
    // Fully inside and fully outside the transitions.
    EXPECT_DOUBLE_EQ(fk_mute_weight(-525.0 * 0.5, 0.5, p), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(-475.0 * 0.5, 0.5, p), 0.0);
    // Cosine value at v = 490 is 0.5 - 0.5 cos(0.3 pi), not 0.3.
    EXPECT_NEAR(fk_mute_weight(-245.0, 0.5, p), 0.5 - 0.5 * std::cos(0.3 * kPi), 1e-12);
}

TEST(FKWeight, FarOutsideBand) {
    FKMuteParams p = event_params();
    // v = 1e6 and v = -1e6 are far from [500, 2000].
    EXPECT_DOUBLE_EQ(fk_mute_weight(-5.0e5, 0.5, p), 1.0);   // Reject keeps it
    EXPECT_DOUBLE_EQ(fk_mute_weight(5.0e5, 0.5, p), 1.0);
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-5.0e5, 0.5, p), 0.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(5.0e5, 0.5, p), 0.0);
}

TEST(FKWeight, ZeroWavenumberIsOutsideAnyFiniteBand) {
    // k = 0, f != 0: v = +inf. Outside any finite band, so Reject keeps it and Pass drops it.
    FKMuteParams p = event_params();
    EXPECT_DOUBLE_EQ(fk_mute_weight(10.0, 0.0, p), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(-10.0, 0.0, p), 1.0);
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(10.0, 0.0, p), 0.0);
    // Even a very wide finite band does not contain v = +inf.
    p.v_min = -1e30;
    p.v_max = 1e30;
    p.taper = 0.0;
    EXPECT_DOUBLE_EQ(fk_mute_weight(10.0, 0.0, p), 0.0);
}

TEST(FKWeight, ZeroFrequencyGivesVelocityZero) {
    // f = 0, k != 0: v = 0.
    FKMuteParams p;                       // [-1500, 1500] contains 0
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.02, p), 0.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, -0.02, p), 0.0);
    FKMuteParams q = event_params();     // [500, 2000] does not
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.02, q), 1.0);
    q.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.02, q), 0.0);
}

TEST(FKWeight, DCWithPreserveDC) {
    FKMuteParams keep;                        // preserve_dc = true (default)
    FKMuteParams keep_pass = keep;
    keep_pass.mode = FKMode::Pass;
    FKMuteParams keep_band = event_params();  // [500, 2000], v = 0 outside
    FKMuteParams keep_band_pass = keep_band;
    keep_band_pass.mode = FKMode::Pass;
    // DC is forced to 1 in both modes, whether or not v = 0 is in the band.
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, keep), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, keep_pass), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, keep_band), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, keep_band_pass), 1.0);
}

TEST(FKWeight, DCWithoutPreserveDC) {
    // Without preserve_dc, DC takes v = 0 like any f = 0 bin.
    FKMuteParams p;
    p.preserve_dc = false;                    // band [-1500, 1500] contains 0
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, p), 0.0);  // Reject mutes it
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, p), 1.0);

    FKMuteParams q = event_params();          // band [500, 2000] does not contain 0
    q.preserve_dc = false;
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, q), 1.0);
    q.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.0, q), 0.0);
}

TEST(FKWeight, HardEdgesWithoutTaper) {
    FKMuteParams p = event_params();
    p.taper = 0.0;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-500.0, 1.0, p), 1.0);       // v = 500: hard edge, outside
    EXPECT_DOUBLE_EQ(fk_mute_weight(-1000.0, 1.0, p), 0.0);      // v = 1000 inside, Reject mutes
    EXPECT_DOUBLE_EQ(fk_mute_weight(-100.0, 1.0, p), 1.0);       // v = 100 outside
    EXPECT_DOUBLE_EQ(fk_mute_weight(-3000.0, 1.0, p), 1.0);      // v = 3000 outside
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-1000.0, 1.0, p), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(-100.0, 1.0, p), 0.0);
}

TEST(FKWeight, SignConventionVelocityIsMinusFOverK) {
    // v = -f / k: (f, k) = (-500, 0.5) and (500, -0.5) both give v = +1000.
    FKMuteParams pos = event_params();
    pos.mode = FKMode::Pass;                  // keeps [500, 2000]
    EXPECT_DOUBLE_EQ(fk_mute_weight(-500.0, 0.5, pos), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(500.0, -0.5, pos), 1.0);
    // (f, k) = (500, 0.5) gives v = -1000: outside [500, 2000].
    EXPECT_DOUBLE_EQ(fk_mute_weight(500.0, 0.5, pos), 0.0);

    FKMuteParams neg = pos;
    neg.v_min = -2000.0;
    neg.v_max = -500.0;
    EXPECT_DOUBLE_EQ(fk_mute_weight(500.0, 0.5, neg), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(-500.0, 0.5, neg), 0.0);
}

TEST(FKWeight, HardEdgeBandIsHalfOpen) {
    // taper = 0: the band is (v_min, v_max]. Pass [500, 2000] excludes v = 500 and includes
    // v = 2000. This follows from section 3 with h = 0: rise(e; e) = 0 by the first branch, so
    // the lower edge is excluded and the upper edge (fall = 1 - rise = 1) is included.
    FKMuteParams p = event_params();
    p.taper = 0.0;
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-500.0, 1.0, p), 0.0);   // v = 500 = v_min
    EXPECT_DOUBLE_EQ(fk_mute_weight(-2000.0, 1.0, p), 1.0);  // v = 2000 = v_max
    p.mode = FKMode::Reject;
    EXPECT_DOUBLE_EQ(fk_mute_weight(-500.0, 1.0, p), 1.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(-2000.0, 1.0, p), 0.0);
}

TEST(FKWeight, NyquistLinesAreNotHermitian) {
    // dt = 2 ms gives F = 250 Hz; dx = 25 m gives K = 0.02 cycles/m. Both are their own
    // negatives on the grid, so W(-f, -k) != W(f, k) there. Section 3 of FK_PIPELINE.md makes
    // this exception explicit: Hermitian symmetry holds everywhere except on the Nyquist row and
    // column. This test pins the weight values only. How the pipeline's inverse treats these bins
    // (it keeps their Hermitian part, as irfft2 does) is checked by
    // MatchesDirectReferenceAtOddSizes, which uses even padded sizes.
    // Pass [5000, 20000]:
    // (F, +K) gives v = -12500 (outside) and (F, -K) gives v = +12500 (inside).
    FKMuteParams p;
    p.dt = 0.002;
    p.dx = 25.0;
    p.v_min = 5000.0;
    p.v_max = 20000.0;
    p.taper = 0.0;
    p.mode = FKMode::Pass;
    EXPECT_DOUBLE_EQ(fk_mute_weight(250.0, 0.02, p), 0.0);
    EXPECT_DOUBLE_EQ(fk_mute_weight(250.0, -0.02, p), 1.0);
}

// Sanity check of the value-negation identity W(f, k) = W(-f, -k). Under v = -f / k this holds
// for every bin, Nyquist lines included, so the test only fails if the sign convention regresses.
// It does not test index-Hermitian behaviour on the Nyquist row and column; that is covered by
// NyquistLinesAreNotHermitian and MatchesDirectReferenceAtOddSizes.
TEST(FKWeight, ValueNegationInvariant) {
    FKMuteParams p = event_params();
    p.mode = FKMode::Pass;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> fd(-125.0, 125.0);
    std::uniform_real_distribution<double> kd(-0.2, 0.2);
    for (int i = 0; i < 2000; ++i) {
        const double f = fd(rng);
        const double k = kd(rng);
        EXPECT_DOUBLE_EQ(fk_mute_weight(f, k, p), fk_mute_weight(-f, -k, p));
    }
    // Including the k = 0 and f = 0 cases.
    EXPECT_DOUBLE_EQ(fk_mute_weight(0.0, 0.1, p), fk_mute_weight(0.0, -0.1, p));
    EXPECT_DOUBLE_EQ(fk_mute_weight(7.0, 0.0, p), fk_mute_weight(-7.0, 0.0, p));
}

TEST(FKWeight, PassRejectComplementary) {
    FKMuteParams reject = event_params();
    FKMuteParams pass = reject;
    pass.mode = FKMode::Pass;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> fd(-125.0, 125.0);
    std::uniform_real_distribution<double> kd(-0.2, 0.2);
    for (int i = 0; i < 2000; ++i) {
        const double f = fd(rng);
        const double k = kd(rng);
        EXPECT_NEAR(fk_mute_weight(f, k, reject) + fk_mute_weight(f, k, pass), 1.0, 1e-12);
    }
}

TEST(FKWeight, InvalidParamsThrow) {
    FKMuteParams p;
    p.dt = 0.0;
    EXPECT_THROW(fk_mute_weight(1.0, 1.0, p), std::invalid_argument);
    p = FKMuteParams{};
    p.v_min = p.v_max;
    EXPECT_THROW(fk_mute_weight(1.0, 1.0, p), std::invalid_argument);
    p = FKMuteParams{};
    p.taper = 1.0;
    EXPECT_THROW(fk_mute_weight(1.0, 1.0, p), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Pipeline: validation, sizes, identities (sections 3-4)
// ---------------------------------------------------------------------------

TEST(FKPipeline, InvalidParamsThrow) {
    const size_t ntr = 8;
    const size_t ns = 16;
    const std::vector<float> in(ntr * ns, 1.0f);
    std::vector<float> out(ntr * ns);
    const FKMuteParams base;

    auto expect_bad = [&](const FKMuteParams& p) {
        EXPECT_THROW(fk_filter_gather(in.data(), out.data(), ntr, ns, p), std::invalid_argument);
    };

    FKMuteParams p = base;
    p.dt = 0.0;
    expect_bad(p);
    p.dt = -0.004;
    expect_bad(p);
    p.dt = std::numeric_limits<double>::quiet_NaN();
    expect_bad(p);

    p = base;
    p.dx = 0.0;
    expect_bad(p);

    p = base;
    p.v_min = 1500.0;
    p.v_max = 1500.0;  // v_min >= v_max
    expect_bad(p);
    p.v_min = 2000.0;
    p.v_max = 1000.0;
    expect_bad(p);

    p = base;
    p.taper = 1.0;
    expect_bad(p);
    p.taper = -0.1;
    expect_bad(p);
    p.taper = std::numeric_limits<double>::quiet_NaN();
    expect_bad(p);

    p = base;
    p.pad_factor = 0.5;
    expect_bad(p);
    p.pad_factor = std::numeric_limits<double>::quiet_NaN();
    expect_bad(p);

    p = base;
    p.mode = static_cast<FKMode>(42);
    expect_bad(p);

    // Dimensions and pointers.
    EXPECT_THROW(fk_filter_gather(in.data(), out.data(), 0, ns, base), std::invalid_argument);
    EXPECT_THROW(fk_filter_gather(in.data(), out.data(), ntr, 0, base), std::invalid_argument);
    EXPECT_THROW(fk_filter_gather(nullptr, out.data(), ntr, ns, base), std::invalid_argument);
    EXPECT_THROW(fk_filter_gather(in.data(), nullptr, ntr, ns, base), std::invalid_argument);

    // Vector overload: size mismatch and zero dimensions.
    EXPECT_THROW(fk_filter_gather(in, ntr + 1, ns, base), std::invalid_argument);
    EXPECT_THROW(fk_filter_gather(in, ntr, 0, base), std::invalid_argument);
    EXPECT_THROW(fk_filter_gather(in, 0, ns, base), std::invalid_argument);

    // Padded size helpers validate too.
    p = base;
    p.pad_factor = 0.5;
    EXPECT_THROW(fk_padded_traces(ntr, p), std::invalid_argument);
    EXPECT_THROW(fk_padded_samples(ns, p), std::invalid_argument);
}

TEST(FKPipeline, PaddedGridOverflowIsRejected) {
    // pad 2^31 on 2 x 2 gives padded axes of 2^32 each, and their product wraps size_t to 0.
    // The pipeline must throw before allocating or indexing with the wrapped size.
    const std::vector<float> in(4, 1.0f);
    std::vector<float> out(4);
    FKMuteParams p;
    p.pad_factor = 2147483648.0;
    EXPECT_THROW(fk_filter_gather(in.data(), out.data(), 2, 2, p), std::invalid_argument);
    EXPECT_THROW(fk_filter_gather(in, 2, 2, p), std::invalid_argument);
}

TEST(FKPipeline, PaddedSizesCoverRequestAndAreSmooth) {
    for (size_t n : {size_t{1}, size_t{2}, size_t{7}, size_t{13}, size_t{100},
                     size_t{1001}, size_t{1500}}) {
        for (double pad : {1.0, 1.5, 2.0, 3.3}) {
            FKMuteParams p;
            p.pad_factor = pad;
            const auto want = static_cast<size_t>(std::ceil(static_cast<double>(n) * pad));
            const size_t pt = fk_padded_traces(n, p);
            const size_t ps = fk_padded_samples(n, p);
            EXPECT_GE(pt, want) << "n=" << n << " pad=" << pad;
            EXPECT_GE(ps, want) << "n=" << n << " pad=" << pad;
            // Complex sizes use 2,3,5,7,11; real sizes use 2,3,5.
            EXPECT_TRUE(only_prime_factors(pt, {2, 3, 5, 7, 11})) << pt;
            EXPECT_TRUE(only_prime_factors(ps, {2, 3, 5})) << ps;
            EXPECT_LE(pt, 2 * want + 16) << "n=" << n << " pad=" << pad;
            EXPECT_LE(ps, 2 * want + 16) << "n=" << n << " pad=" << pad;
            // Minimal, not just smooth: no smooth size lies in [want, padded).
            for (size_t m = want; m < pt; ++m) {
                EXPECT_FALSE(only_prime_factors(m, {2, 3, 5, 7, 11}))
                    << "n=" << n << " pad=" << pad << " m=" << m;
            }
            for (size_t m = want; m < ps; ++m) {
                EXPECT_FALSE(only_prime_factors(m, {2, 3, 5}))
                    << "n=" << n << " pad=" << pad << " m=" << m;
            }
        }
    }
    // Default pad factor 1.5: 1001 * 1.5 = 1501.5 -> at least 1502 samples.
    FKMuteParams p;
    EXPECT_GE(fk_padded_samples(1001, p), 1502u);

    // An absurd pad factor is rejected instead of overflowing.
    p.pad_factor = 1e300;
    EXPECT_THROW(fk_padded_traces(1001, p), std::invalid_argument);
    EXPECT_THROW(fk_padded_samples(1001, p), std::invalid_argument);
}

TEST(FKPipeline, EdgeTaperFormulaWithEmptyBand) {
    // A band far above any velocity in the data leaves every bin with weight 1 under Reject,
    // so the output is exactly the tapered input.
    const size_t ntr = 30;
    const size_t ns = 80;
    const auto in = random_gather(ntr, ns, 3);
    FKMuteParams p;
    p.v_min = 1e6;
    p.v_max = 1e6 + 1.0;
    p.taper = 0.0;
    p.edge_taper_traces = 4;
    p.edge_taper_samples = 3;
    const auto out = fk_filter_gather(in, ntr, ns, p);

    std::vector<float> want(ntr * ns);
    for (size_t t = 0; t < ntr; ++t) {
        double wt = 1.0;
        if (t < 4) {
            wt *= taper_weight(t, 4);
        }
        if (ntr - 1 - t < 4) {
            wt *= taper_weight(ntr - 1 - t, 4);
        }
        for (size_t s = 0; s < ns; ++s) {
            double ws = 1.0;
            if (s < 3) {
                ws *= taper_weight(s, 3);
            }
            if (ns - 1 - s < 3) {
                ws *= taper_weight(ns - 1 - s, 3);
            }
            want[t * ns + s] = static_cast<float>(static_cast<double>(in[t * ns + s]) * wt * ws);
        }
    }
    EXPECT_LE(rel_rms(out, want), 1e-5);
}

TEST(FKPipeline, EdgeTapersMultiplyWhenTheyOverlap) {
    // One trace is reached by both the start and the end taper. Section 4, step 3 defines each
    // taper but does not say how two overlapping tapers combine; this test pins the product,
    // which is the behaviour of the implementation. The weight is w(0)^2 with N = 4.
    // w(0) = 0.5 - 0.5 cos(pi / 5), so w(0)^2 = 0.00911863.
    // An empty Reject band leaves the tapered input as the output.
    const size_t ntr = 1;
    const size_t ns = 9;
    const auto in = random_gather(ntr, ns, 29);
    FKMuteParams p;
    p.v_min = 1e6;
    p.v_max = 1e6 + 1.0;
    p.taper = 0.0;
    p.edge_taper_traces = 4;
    p.edge_taper_samples = 0;
    const auto out = fk_filter_gather(in, ntr, ns, p);

    const double w = taper_weight(0, 4) * taper_weight(0, 4);
    EXPECT_NEAR(w, 0.00911863, 1e-8);
    std::vector<float> want(ns);
    for (size_t s = 0; s < ns; ++s) {
        want[s] = static_cast<float>(static_cast<double>(in[s]) * w);
    }
    EXPECT_LE(rel_rms(out, want), 1e-5);
}

TEST(FKPipeline, EdgeTaperSizeMaxStaysFinite) {
    // n + 1 used to wrap to 0 for n = SIZE_MAX, which gave NaN taper weights.
    const size_t ntr = 4;
    const size_t ns = 8;
    const std::vector<float> in(ntr * ns, 1.0f);
    const size_t big = std::numeric_limits<size_t>::max();
    FKMuteParams p;
    p.v_min = 1e6;
    p.v_max = 1e6 + 1.0;
    p.taper = 0.0;

    p.edge_taper_traces = big;
    p.edge_taper_samples = 0;
    for (float v : fk_filter_gather(in, ntr, ns, p)) {
        EXPECT_TRUE(std::isfinite(v));
    }
    p.edge_taper_traces = 0;
    p.edge_taper_samples = big;
    for (float v : fk_filter_gather(in, ntr, ns, p)) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(FKPipeline, RejectEmptyBandKeepsInteriorTraces) {
    // Events at +-1000 m/s. The band [-400, 400] lies between them, so it holds no event
    // energy. Section 4 applies the 1e-4 empty-band criterion to smooth input, meaning events
    // with a Hann envelope along x. This test uses that input. Hard-truncated events leak into
    // every band and are covered by RejectEmptyBandHardEdgedLeakage.
    const auto in = two_event_gather(true, true, true);
    FKMuteParams p = event_params();
    p.v_min = -400.0;
    p.v_max = 400.0;
    const auto out = fk_filter_gather(in, kEvNtr, kEvNs, p);
    const auto out_mid = trace_rows(out, kEvNs, 4, kEvNtr - 4);
    const auto in_mid = trace_rows(in, kEvNs, 4, kEvNtr - 4);
    EXPECT_LE(rel_rms(out_mid, in_mid), 1e-4);
}

TEST(FKPipeline, RejectEmptyBandHardEdgedLeakage) {
    // Hard-edged +-1000 m/s events, Reject of the empty band [-400, 400], taper 0.1 and the
    // default 4-trace edge taper. Section 4 limits the 1e-4 empty-band criterion to smooth input,
    // so hard-truncated events are expected to exceed 1e-4 here. The leakage comes from the
    // finite aperture of the truncated events in f-k space, which spreads energy into every band.
    // The measured value is 5.7e-3. The 1e-2 bound is a regression guard: it fails if the
    // leakage grows well beyond the measured level.
    const auto in = two_event_gather(true, true, false);
    FKMuteParams p = event_params();
    p.v_min = -400.0;
    p.v_max = 400.0;
    const auto out = fk_filter_gather(in, kEvNtr, kEvNs, p);
    const auto out_mid = trace_rows(out, kEvNs, 4, kEvNtr - 4);
    const auto in_mid = trace_rows(in, kEvNs, 4, kEvNtr - 4);
    EXPECT_LE(rel_rms(out_mid, in_mid), 1e-2);
    // Section 4: hard-edged input is not expected to meet the 1e-4 empty-band criterion.
    EXPECT_GT(rel_rms(out_mid, in_mid), 1e-4);
}

TEST(FKPipeline, IdentityConfigReturnsRawInput) {
    // Section 4 identity configuration: Reject with a band that no bin falls in (v_min = 1e29,
    // v_max = 1e30), taper 0 and no edge tapers. Every bin keeps weight 1, so the raw input comes
    // back to 1e-5 relative RMS. The input is not made zero-mean, so the k = 0 row is included.
    // The sizes cover a small non-square gather and the realistic 1001 x 1500 case.
    FKMuteParams p;
    p.v_min = 1e29;
    p.v_max = 1e30;
    p.taper = 0.0;
    p.mode = FKMode::Reject;
    p.edge_taper_traces = 0;
    p.edge_taper_samples = 0;
    for (const auto& dims :
         {std::pair<size_t, size_t>{40, 128}, std::pair<size_t, size_t>{1001, 1500}}) {
        const size_t ntr = dims.first;
        const size_t ns = dims.second;
        const auto in = random_gather(ntr, ns, 61);
        const auto out = fk_filter_gather(in, ntr, ns, p);
        EXPECT_LE(rel_rms(out, in), 1e-5) << "ntr=" << ntr << " ns=" << ns;
    }
}

TEST(FKPipeline, AllPassZeroMeanInputPassesThrough) {
    // Pass with a huge band is intended to drop the k = 0 row. Section 2 gives that row
    // v = +inf, which lies outside every finite band, so Pass removes the trace-to-trace mean.
    // To isolate the other bins, the input is made zero-mean across traces first, so the dropped
    // row is already zero and the output should equal the input. The raw-input behaviour is
    // checked by AllPassOnRawInputRemovesOnlyTraceMean.
    const size_t ntr = 40;
    const size_t ns = 128;
    auto in = random_gather(ntr, ns, 5);
    remove_trace_mean(in, ntr, ns);
    FKMuteParams p;
    p.v_min = -1e30;
    p.v_max = 1e30;
    p.taper = 0.0;
    p.mode = FKMode::Pass;
    p.edge_taper_traces = 0;
    p.edge_taper_samples = 0;
    const auto out = fk_filter_gather(in, ntr, ns, p);
    EXPECT_LE(rel_rms(out, in), 1e-5);
}

TEST(FKPipeline, AllPassOnRawInputRemovesOnlyTraceMean) {
    // Pass with a huge band on raw (not zero-mean) input. The k = 0, f != 0 row has v = +inf,
    // which lies outside every finite band, so Pass removes it by design (section 2). What
    // remains is the trace-mean component plus the DC bin, which preserve_dc keeps:
    //   out(t, s) = in(t, s) - m(s) + M,
    // where m(s) = sum_t in(t, s) / ntr_pad and M = sum(in) / (ntr_pad * ns_pad). Both sums run
    // over the valid block, and the zero padding counts in the normalisation.
    // The identity configuration that returns raw input is covered by
    // IdentityConfigReturnsRawInput.
    const size_t ntr = 40;
    const size_t ns = 128;
    const auto in = random_gather(ntr, ns, 5);
    FKMuteParams p;
    p.v_min = -1e30;
    p.v_max = 1e30;
    p.taper = 0.0;
    p.mode = FKMode::Pass;
    p.edge_taper_traces = 0;
    p.edge_taper_samples = 0;
    const auto out = fk_filter_gather(in, ntr, ns, p);

    const double ntr_pad = static_cast<double>(fk_padded_traces(ntr, p));
    const double ns_pad = static_cast<double>(fk_padded_samples(ns, p));
    std::vector<double> m(ns, 0.0);
    double total = 0.0;
    for (size_t t = 0; t < ntr; ++t) {
        for (size_t s = 0; s < ns; ++s) {
            m[s] += in[t * ns + s];
            total += in[t * ns + s];
        }
    }
    for (double& ms : m) {
        ms /= ntr_pad;
    }
    const double mean_all = total / (ntr_pad * ns_pad);
    std::vector<float> want(ntr * ns);
    for (size_t t = 0; t < ntr; ++t) {
        for (size_t s = 0; s < ns; ++s) {
            want[t * ns + s] = static_cast<float>(in[t * ns + s] - m[s] + mean_all);
        }
    }
    EXPECT_LE(rel_rms(out, want), 1e-5);
    // Guard against the test silently passing as an identity check: the trace mean is not zero.
    EXPECT_GT(rel_rms(out, in), 1e-2);
}

TEST(FKPipeline, PreserveDCControlsOffsetWithEmptyBand) {
    // With an empty Pass band only the DC bin can survive, so the output is the constant
    // sum(in) / (ntr_pad * ns_pad) when preserve_dc is true, and zero otherwise.
    const size_t ntr = 30;
    const size_t ns = 80;
    const auto in = random_gather(ntr, ns, 17);
    double sum = 0.0;
    for (float v : in) {
        sum += v;
    }
    FKMuteParams p;
    p.v_min = 1e6;
    p.v_max = 1e6 + 1.0;
    p.taper = 0.0;
    p.mode = FKMode::Pass;
    p.edge_taper_traces = 0;
    p.edge_taper_samples = 0;

    const double scale = static_cast<double>(fk_padded_traces(ntr, p)) *
                         static_cast<double>(fk_padded_samples(ns, p));
    p.preserve_dc = true;
    const auto kept = fk_filter_gather(in, ntr, ns, p);
    const std::vector<float> want(kept.size(), static_cast<float>(sum / scale));
    EXPECT_LE(rel_rms(kept, want), 1e-5);

    p.preserve_dc = false;
    const auto dropped = fk_filter_gather(in, ntr, ns, p);
    for (float v : dropped) {
        EXPECT_NEAR(v, 0.0f, 1e-6f);
    }
}

TEST(FKPipeline, SignConventionPositiveVelocityKeepsEvent) {
    // Event A (+1000 m/s) has arrival increasing with x, so its apparent velocity is +1000.
    const auto in = two_event_gather(true, true);

    FKMuteParams pass_pos = event_params();
    pass_pos.mode = FKMode::Pass;  // keep [500, 2000]
    const auto out_pos = fk_filter_gather(in, kEvNtr, kEvNs, pass_pos);
    EXPECT_GE(retained_db(out_pos, in, kEventA), -1.0);
    EXPECT_LE(retained_db(out_pos, in, kEventB), -20.0);

    FKMuteParams pass_neg = pass_pos;
    pass_neg.v_min = -2000.0;
    pass_neg.v_max = -500.0;       // keep [-2000, -500]
    const auto out_neg = fk_filter_gather(in, kEvNtr, kEvNs, pass_neg);
    EXPECT_LE(retained_db(out_neg, in, kEventA), -20.0);
    EXPECT_GE(retained_db(out_neg, in, kEventB), -1.0);
}

TEST(FKPipeline, RejectAttenuatesPositiveEventOnly) {
    const auto in = two_event_gather(true, true);
    const auto out = fk_filter_gather(in, kEvNtr, kEvNs, event_params());  // Reject [500, 2000]
    EXPECT_LE(retained_db(out, in, kEventA), -20.0);
    EXPECT_GE(retained_db(out, in, kEventB), -1.0);
}

TEST(FKPipeline, PassAndRejectSumToInput) {
    // Pass = inside, Reject = 1 - inside, so the two outputs add up to the input.
    // preserve_dc is off, because a forced DC weight would count twice.
    const size_t ntr = 30;
    const size_t ns = 80;
    const auto in = random_gather(ntr, ns, 23);
    FKMuteParams reject = event_params();
    reject.preserve_dc = false;
    reject.edge_taper_traces = 0;
    FKMuteParams pass = reject;
    pass.mode = FKMode::Pass;
    const auto r = fk_filter_gather(in, ntr, ns, reject);
    const auto q = fk_filter_gather(in, ntr, ns, pass);
    std::vector<float> sum(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        sum[i] = r[i] + q[i];
    }
    EXPECT_LE(rel_rms(sum, in), 1e-5);
}

TEST(FKPipeline, MatchesDirectReferenceAtOddSizes) {
    // Odd and non-square gathers, with an asymmetric band, taper and edge tapers. Even padded
    // trace counts (e.g. 9 -> 14) exercise the k-Nyquist row, where the mute weight is not
    // Hermitian-symmetric (section 3, Nyquist note).
    struct Case {
        size_t ntr;
        size_t ns;
        FKMode mode;
        size_t edge_traces;
        size_t edge_samples;
    };
    const Case cases[] = {
        {7, 13, FKMode::Reject, 4, 0},
        {7, 13, FKMode::Pass, 3, 5},
        {9, 33, FKMode::Reject, 3, 5},
        {9, 33, FKMode::Pass, 0, 0},
        {13, 7, FKMode::Reject, 4, 3},
        {1, 1, FKMode::Pass, 0, 0},
    };
    unsigned seed = 71;
    for (const Case& c : cases) {
        const auto in = random_gather(c.ntr, c.ns, seed++);
        FKMuteParams p;
        p.dt = 0.002;
        p.dx = 5.0;
        p.v_min = -800.0;
        p.v_max = 2500.0;
        p.taper = 0.2;
        p.mode = c.mode;
        p.edge_taper_traces = c.edge_traces;
        p.edge_taper_samples = c.edge_samples;
        const auto out = fk_filter_gather(in, c.ntr, c.ns, p);
        const auto ref = direct_fk_reference(in, c.ntr, c.ns, p);
        EXPECT_LE(rel_rms(out, ref), 1e-5)
            << "ntr=" << c.ntr << " ns=" << c.ns << " mode=" << static_cast<int>(c.mode);
    }
}

TEST(FKPipeline, AllPassZeroMeanInputPassesThroughAtLargeOddSize) {
    // 1001 x 1500 pads to non-power-of-two sizes. Zero-mean input (k = 0 row removed first) must
    // come back unchanged under the Pass configuration with a huge band. This checks the padding
    // offsets at a realistic size. The k = 0 row is dropped by design (see
    // AllPassZeroMeanInputPassesThrough).
    const size_t ntr = 1001;
    const size_t ns = 1500;
    auto in = random_gather(ntr, ns, 67);
    remove_trace_mean(in, ntr, ns);
    FKMuteParams p;
    p.v_min = -1e30;
    p.v_max = 1e30;
    p.taper = 0.0;
    p.mode = FKMode::Pass;
    p.edge_taper_traces = 0;
    p.edge_taper_samples = 0;
    const auto out = fk_filter_gather(in, ntr, ns, p);
    EXPECT_LE(rel_rms(out, in), 1e-5);
}

TEST(FKPipeline, OutputsAreFiniteForOddSizes) {
    const FKMuteParams p;  // defaults
    const auto small_in = random_gather(7, 13, 31);
    const auto small = fk_filter_gather(small_in, 7, 13, p);
    ASSERT_EQ(small.size(), 7u * 13u);
    for (float v : small) {
        EXPECT_TRUE(std::isfinite(v));
    }

    const auto big_in = random_gather(1001, 1500, 37);
    const auto big = fk_filter_gather(big_in, 1001, 1500, p);
    ASSERT_EQ(big.size(), 1001u * 1500u);
    for (float v : big) {
        ASSERT_TRUE(std::isfinite(v));
    }
}

TEST(FKPipeline, ThreadsTwoMatchesOne) {
    const size_t ntr = 257;
    const size_t ns = 600;
    const auto in = random_gather(ntr, ns, 41);
    FKMuteParams p1 = event_params();
    p1.threads = 1;
    FKMuteParams p2 = p1;
    p2.threads = 2;
    const auto a = fk_filter_gather(in, ntr, ns, p1);
    const auto b = fk_filter_gather(in, ntr, ns, p2);
    ASSERT_EQ(a.size(), b.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LE(max_diff, 1e-6);
}

TEST(FKPipeline, ThreadsZeroUsesHardwareConcurrency) {
    const size_t ntr = 64;
    const size_t ns = 100;
    const auto in = random_gather(ntr, ns, 43);
    FKMuteParams p1 = event_params();
    p1.threads = 1;
    FKMuteParams p0 = p1;
    p0.threads = 0;
    const auto a = fk_filter_gather(in, ntr, ns, p1);
    const auto b = fk_filter_gather(in, ntr, ns, p0);
    EXPECT_LE(rel_rms(b, a), 1e-6);
}

TEST(FKPipeline, InPlaceAliasingMatchesOutOfPlace) {
    const size_t ntr = 50;
    const size_t ns = 120;
    const auto in = random_gather(ntr, ns, 47);
    const FKMuteParams p = event_params();
    const auto expected = fk_filter_gather(in, ntr, ns, p);

    std::vector<float> buf = in;
    fk_filter_gather(buf.data(), buf.data(), ntr, ns, p);
    EXPECT_EQ(buf, expected);
}

TEST(FKPipeline, VectorOverloadMatchesPointerOverload) {
    const size_t ntr = 33;
    const size_t ns = 47;
    const auto in = random_gather(ntr, ns, 53);
    const FKMuteParams p = event_params();
    std::vector<float> out(in.size());
    fk_filter_gather(in.data(), out.data(), ntr, ns, p);
    const auto via_vec = fk_filter_gather(in, ntr, ns, p);
    EXPECT_EQ(via_vec, out);
}

TEST(FKPipeline, ParallelWeightTableAndMultiplyMatchSerial) {
    // Large enough that the weight-table build and the weight multiply get four workers, not
    // two: each worker needs about 256k spectrum elements, and 1536 x 1126 is about 1.7M.
    const size_t ntr = 1001;
    const size_t ns = 1500;
    const auto in = random_gather(ntr, ns, 61);
    FKMuteParams p1 = event_params();
    p1.threads = 1;
    p1.mode = FKMode::Pass;
    FKMuteParams p4 = p1;
    p4.threads = 4;
    const size_t spec_count = fk_padded_traces(ntr, p1) * (fk_padded_samples(ns, p1) / 2 + 1);
    ASSERT_GE(spec_count, size_t{4} * 256 * 1024);
    const auto a = fk_filter_gather(in, ntr, ns, p1);
    const auto b = fk_filter_gather(in, ntr, ns, p4);
    ASSERT_EQ(a.size(), b.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    EXPECT_LE(max_diff, 1e-6);
}

// Runs fk_filter_gather on a new thread, whose thread_local weight cache starts empty. The result
// is a reference that cannot come from a cache hit.
std::vector<float> filter_on_fresh_thread(const std::vector<float>& in, size_t ntr, size_t ns,
                                          const FKMuteParams& p) {
    std::vector<float> out;
    std::thread worker([&] { out = fk_filter_gather(in, ntr, ns, p); });
    worker.join();
    return out;
}

TEST(FKPipeline, WeightCacheKeyFieldsAreNotStale) {
    // Each step changes one key field (or returns to an earlier configuration, which should hit
    // the cache). Every in-thread result must equal a result computed on a fresh thread.
    const size_t ntr = 40;
    const size_t ns = 90;
    const auto in = random_gather(ntr, ns, 59);
    const FKMuteParams base = event_params();
    FKMuteParams dt = base;
    dt.dt = 0.003;
    FKMuteParams dx = base;
    dx.dx = 10.0;
    FKMuteParams taper = base;
    taper.taper = 0.5;
    FKMuteParams vmin = base;
    vmin.v_min = 300.0;
    FKMuteParams vmax = base;
    vmax.v_max = 2500.0;
    FKMuteParams pass = base;
    pass.mode = FKMode::Pass;
    FKMuteParams pass_no_dc = pass;
    pass_no_dc.preserve_dc = false;

    const std::vector<std::pair<const char*, FKMuteParams>> steps = {
        {"base", base}, {"dt", dt}, {"base", base}, {"dx", dx}, {"base", base},
        {"taper", taper}, {"base", base}, {"v_min", vmin}, {"base", base}, {"v_max", vmax},
        {"pass", pass}, {"pass_no_dc", pass_no_dc}, {"pass", pass}, {"base", base},
    };
    const auto base_ref = filter_on_fresh_thread(in, ntr, ns, base);
    for (const auto& step : steps) {
        const auto cached = fk_filter_gather(in, ntr, ns, step.second);
        const auto fresh = filter_on_fresh_thread(in, ntr, ns, step.second);
        EXPECT_EQ(cached, fresh) << "step " << step.first;
    }
    // Sanity check that the steps are not trivially equal to the base configuration.
    for (const auto& step : {dt, dx, taper, vmin, vmax, pass}) {
        EXPECT_GT(rel_rms(filter_on_fresh_thread(in, ntr, ns, step), base_ref), 1e-6);
    }
}

TEST(FKPipeline, ReusedThreadBuffersLeaveNoStalePadding) {
    // A large gather fills the thread_local work buffers with data in a different layout. A
    // smaller gather on the same thread must not pick up any of it through the padding.
    const size_t big_ntr = 300;
    const size_t big_ns = 500;
    const auto big_in = random_gather(big_ntr, big_ns, 67);
    const size_t ntr = 40;
    const size_t ns = 90;
    const auto small_in = random_gather(ntr, ns, 71);
    FKMuteParams p = event_params();
    p.edge_taper_samples = 5;
    const auto fresh = filter_on_fresh_thread(small_in, ntr, ns, p);
    std::vector<float> reused;
    std::thread worker([&] {
        fk_filter_gather(big_in, big_ntr, big_ns, p);
        reused = fk_filter_gather(small_in, ntr, ns, p);
    });
    worker.join();
    EXPECT_EQ(reused, fresh);
}
