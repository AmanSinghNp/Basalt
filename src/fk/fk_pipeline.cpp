// F-K mute pipeline: edge tapers, 2D real-to-complex FFT (pocketfft), apparent-
// velocity band weighting and complex-to-real inverse. Contract: docs/design/FK_PIPELINE.md.

#include "basalt/fk.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "pocketfft_hdronly.h"

namespace basalt {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Band geometry precomputed once per call so the per-bin work is only arithmetic
// plus a cosine inside the raised-cosine transition zones.
struct BandCoeffs {
    double v_min;
    double v_max;
    double h_min;       // transition half width at v_min
    double h_max;       // transition half width at v_max
    bool reject;
    bool preserve_dc;
};

void validate_params(const FKMuteParams& p) {
    if (!(std::isfinite(p.dt) && p.dt > 0.0)) {
        throw std::invalid_argument("fk: dt must be finite and > 0");
    }
    if (!(std::isfinite(p.dx) && p.dx > 0.0)) {
        throw std::invalid_argument("fk: dx must be finite and > 0");
    }
    if (!(std::isfinite(p.v_min) && std::isfinite(p.v_max) && p.v_min < p.v_max)) {
        throw std::invalid_argument("fk: v_min must be finite and < v_max");
    }
    if (!(p.taper >= 0.0 && p.taper < 1.0)) {
        throw std::invalid_argument("fk: taper must be in [0, 1)");
    }
    if (!(std::isfinite(p.pad_factor) && p.pad_factor >= 1.0)) {
        throw std::invalid_argument("fk: pad_factor must be finite and >= 1");
    }
    if (p.mode != FKMode::Reject && p.mode != FKMode::Pass) {
        throw std::invalid_argument("fk: unknown mode");
    }
}

void validate_dims(size_t ntraces, size_t nsamples) {
    if (ntraces == 0) {
        throw std::invalid_argument("fk: ntraces must be > 0");
    }
    if (nsamples == 0) {
        throw std::invalid_argument("fk: nsamples must be > 0");
    }
}

BandCoeffs make_band(const FKMuteParams& p) {
    BandCoeffs b{};
    b.v_min = p.v_min;
    b.v_max = p.v_max;
    b.h_min = 0.5 * p.taper * std::fabs(p.v_min);
    b.h_max = 0.5 * p.taper * std::fabs(p.v_max);
    b.reject = (p.mode == FKMode::Reject);
    b.preserve_dc = p.preserve_dc;
    return b;
}

// Raised-cosine step from 0 to 1 centred on edge `e` with half width `h`.
// With h == 0 the edge is hard (the v <= e branch wins at v == e).
inline double rise(double v, double e, double h) {
    if (v <= e - h) {
        return 0.0;
    }
    if (v >= e + h) {
        return 1.0;
    }
    return 0.5 - 0.5 * std::cos(kPi * (v - (e - h)) / (2.0 * h));
}

// Membership of apparent velocity v in [v_min, v_max], smoothed at both edges.
inline double band_inside(double v, const BandCoeffs& b) {
    return rise(v, b.v_min, b.h_min) * (1.0 - rise(v, b.v_max, b.h_max));
}

// Mute weight of one f-k bin. Frequency in Hz, wavenumber in cycles per metre.
inline double bin_weight(double f_hz, double k_cpm, const BandCoeffs& b) {
    double inside;
    if (f_hz == 0.0 && k_cpm == 0.0) {
        if (b.preserve_dc) {
            return 1.0;
        }
        // -f / k is 0 / 0 here. The DC bin is evaluated at v = 0 (documented in fk.hpp).
        inside = band_inside(0.0, b);
    } else if (k_cpm == 0.0) {
        // Vertical / flat event: v = +inf, outside any finite band.
        inside = band_inside(std::numeric_limits<double>::infinity(), b);
    } else {
        inside = band_inside(-f_hz / k_cpm, b);
    }
    return b.reject ? 1.0 - inside : inside;
}

// Raised-cosine taper weight for the i-th sample from an edge, i = 0 .. n-1. n + 1 is formed
// as a double so that n = SIZE_MAX cannot wrap to zero (which would give NaN).
inline double edge_taper_weight(size_t i, size_t n) {
    return 0.5 - 0.5 * std::cos(kPi * static_cast<double>(i + 1) / (static_cast<double>(n) + 1.0));
}

// Product of the start and end edge tapers along a line of length `len`. When the two tapers
// reach the same sample (len < 2n) their weights multiply. Section 4, step 3 of
// FK_PIPELINE.md defines each taper and does not cover the overlap, so the product is the
// behaviour of this implementation.
std::vector<double> make_edge_taper(size_t len, size_t n) {
    std::vector<double> w(len, 1.0);
    const size_t m = std::min(n, len);
    for (size_t i = 0; i < m; ++i) {
        const double we = edge_taper_weight(i, n);
        w[i] *= we;
        w[len - 1 - i] *= we;
    }
    return w;
}

size_t resolve_threads(size_t requested) {
    if (requested != 0) {
        return requested;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<size_t>(hc);
}

// ceil(n * pad_factor) as a size. Rejects values that would not fit (and whose FFT could
// never be allocated) before the conversion, which would otherwise be undefined.
size_t padded_length(size_t n, double pad_factor) {
    const double padded = std::ceil(static_cast<double>(n) * pad_factor);
    if (!(padded < 9.0e15)) {
        throw std::invalid_argument("fk: padded size is too large");
    }
    return static_cast<size_t>(padded);
}

} // namespace

size_t fk_padded_traces(size_t ntraces, const FKMuteParams& params) {
    validate_params(params);
    if (ntraces == 0) {
        return 0;
    }
    return pocketfft::detail::util::good_size_cmplx(padded_length(ntraces, params.pad_factor));
}

size_t fk_padded_samples(size_t nsamples, const FKMuteParams& params) {
    validate_params(params);
    if (nsamples == 0) {
        return 0;
    }
    return pocketfft::detail::util::good_size_real(padded_length(nsamples, params.pad_factor));
}

double fk_mute_weight(double f_hz, double k_cpm, const FKMuteParams& params) {
    validate_params(params);
    return bin_weight(f_hz, k_cpm, make_band(params));
}

void fk_filter_gather(const float* in, float* out,
                      size_t ntraces, size_t nsamples,
                      const FKMuteParams& params) {
    validate_params(params);
    validate_dims(ntraces, nsamples);
    if (in == nullptr || out == nullptr) {
        throw std::invalid_argument("fk: null input or output pointer");
    }

    const size_t ntr_pad = fk_padded_traces(ntraces, params);
    const size_t ns_pad = fk_padded_samples(nsamples, params);
    const size_t nfreq = ns_pad / 2 + 1;
    // The padded real grid holds ntr_pad * ns_pad elements and the spectrum ntr_pad * nfreq,
    // with nfreq <= ns_pad. Reject a product that would wrap size_t before anything indexes
    // or allocates with it. ns_pad >= 1, so the division is safe.
    if (ntr_pad > std::numeric_limits<size_t>::max() / ns_pad) {
        throw std::invalid_argument("fk: padded grid is too large");
    }
    const size_t nthreads = resolve_threads(params.threads);

    // Step 3: copy the input into the top-left of a zeroed padded buffer, with edge tapers.
    const std::vector<double> wt = make_edge_taper(ntraces, params.edge_taper_traces);
    const std::vector<double> ws = make_edge_taper(nsamples, params.edge_taper_samples);
    std::vector<float> real(ntr_pad * ns_pad, 0.0f);
    for (size_t t = 0; t < ntraces; ++t) {
        const float* src = in + t * nsamples;
        float* dst = real.data() + t * ns_pad;
        for (size_t s = 0; s < nsamples; ++s) {
            dst[s] = static_cast<float>(static_cast<double>(src[s]) * wt[t] * ws[s]);
        }
    }

    // Step 4: forward real-to-complex 2D FFT over {traces, samples}. Strides in bytes.
    const pocketfft::shape_t shape{ntr_pad, ns_pad};
    const pocketfft::shape_t axes{0, 1};
    const pocketfft::stride_t stride_real{
        static_cast<std::ptrdiff_t>(ns_pad * sizeof(float)),
        static_cast<std::ptrdiff_t>(sizeof(float))};
    const pocketfft::stride_t stride_spec{
        static_cast<std::ptrdiff_t>(nfreq * sizeof(std::complex<float>)),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};

    std::vector<std::complex<float>> spec(ntr_pad * nfreq);
    pocketfft::r2c<float>(shape, stride_real, stride_spec, axes, true,
                          real.data(), spec.data(), 1.0f, nthreads);

    // Step 5: weight every bin. f_j = j / (ns_pad dt); k_i = fftfreq(ntr_pad, dx)[i]
    // using numpy's convention (indices above (n - 1) / 2 are negative).
    // Note: W(-f, -k) = W(f, k) holds everywhere except on the Nyquist row and column, which
    // are their own mirror images (FK_PIPELINE.md section 3, Nyquist note). For even ntr_pad,
    // the k-Nyquist row has k = -K, and v(f, -K) = f / K is not v(-f, -K). For even ns_pad, the
    // f-Nyquist column has f = +F, and +F and -F are the same bin, so v flips sign with k and
    // W(F, k) != W(F, -k) in general. Both cases only matter for aliased data (|v| >= dx / dt).
    // These Nyquist bins are not special-cased. The c2r inverse below keeps only their
    // Hermitian part, which matches numpy.fft.irfft2(rfft2(x) * w). A Hermitian-symmetrised
    // filter gives a different result. The difference depends on the band, the grid size and
    // the input, and is not a small rounding effect: on random input it reached about 23%
    // relRMS for 9 x 33 gathers with a Pass band of [500, 2000] m/s (dt = 2 ms, dx = 5 m).
    const BandCoeffs band = make_band(params);
    const double df = 1.0 / (static_cast<double>(ns_pad) * params.dt);
    const double dk = 1.0 / (static_cast<double>(ntr_pad) * params.dx);
    const size_t k_pos_max = (ntr_pad - 1) / 2;
    for (size_t i = 0; i < ntr_pad; ++i) {
        const double k_index = (i <= k_pos_max)
            ? static_cast<double>(i)
            : static_cast<double>(i) - static_cast<double>(ntr_pad);
        const double k_cpm = k_index * dk;
        std::complex<float>* row = spec.data() + i * nfreq;
        for (size_t j = 0; j < nfreq; ++j) {
            const double f_hz = static_cast<double>(j) * df;
            row[j] *= static_cast<float>(bin_weight(f_hz, k_cpm, band));
        }
    }

    // Step 6: complex-to-real inverse 2D FFT with scale 1 / (ntr_pad * ns_pad).
    // The padded buffer is free once the forward transform is done; reuse it for the result.
    const float inv_scale = static_cast<float>(
        1.0 / (static_cast<double>(ntr_pad) * static_cast<double>(ns_pad)));
    pocketfft::c2r<float>(shape, stride_spec, stride_real, axes, false,
                          spec.data(), real.data(), inv_scale, nthreads);

    // Step 7: copy the valid top-left block out. `in` is no longer read, so aliasing is safe.
    for (size_t t = 0; t < ntraces; ++t) {
        const float* src = real.data() + t * ns_pad;
        float* dst = out + t * nsamples;
        std::copy(src, src + nsamples, dst);
    }
}

std::vector<float> fk_filter_gather(const std::vector<float>& in,
                                    size_t ntraces, size_t nsamples,
                                    const FKMuteParams& params) {
    validate_dims(ntraces, nsamples);
    if (in.size() / nsamples != ntraces || in.size() % nsamples != 0) {
        throw std::invalid_argument("fk: input size does not match ntraces * nsamples");
    }
    std::vector<float> out(in.size());
    fk_filter_gather(in.data(), out.data(), ntraces, nsamples, params);
    return out;
}

} // namespace basalt
