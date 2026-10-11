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

// Passes touching at least this many elements are split across worker threads. Smaller passes
// run on the calling thread so that small gathers do not pay for thread creation.
constexpr size_t kParallelMinWork = size_t{256} * 1024;

// Weight tables kept per calling thread. Two covers alternating between two configurations.
constexpr size_t kWeightCacheEntries = 2;

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

// Joins every joinable thread in the list when it goes out of scope. Destroying a joinable
// std::thread calls std::terminate, so this keeps unwinding (e.g. from the calling thread's own
// share of the work) from reaching that state.
class ThreadJoiner {
public:
    explicit ThreadJoiner(std::vector<std::thread>& threads) : threads_(threads) {}
    ThreadJoiner(const ThreadJoiner&) = delete;
    ThreadJoiner& operator=(const ThreadJoiner&) = delete;
    ~ThreadJoiner() {
        for (std::thread& t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

private:
    std::vector<std::thread>& threads_;
};

// Calls fn(begin, end) on disjoint, contiguous row ranges that together cover [0, nrows).
// The rows are split statically across up to `threads` threads, the calling thread included.
// Workers are only started when `work` (elements touched) reaches kParallelMinWork, and the
// thread count is capped so that each worker gets about kParallelMinWork elements. A pass just
// over the cutoff therefore uses two threads, not `threads`. fn may throw on the calling thread,
// and the exception propagates after the workers are joined. fn must not throw on a worker
// thread, because that would call std::terminate.
template <typename Fn>
void for_each_row_range(size_t nrows, size_t threads, size_t work, Fn&& fn) {
    const size_t by_work = std::max<size_t>(2, work / kParallelMinWork);
    const size_t nt = std::min({threads, nrows, by_work});
    if (nt <= 1 || work < kParallelMinWork) {
        fn(size_t{0}, nrows);
        return;
    }
    const size_t chunk = (nrows + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(nt - 1);
    ThreadJoiner joiner(workers);
    for (size_t w = 1; w < nt; ++w) {
        const size_t begin = std::min(nrows, w * chunk);
        const size_t end = std::min(nrows, begin + chunk);
        if (begin == end) {
            break;
        }
        try {
            workers.emplace_back([&fn, begin, end] { fn(begin, end); });
        } catch (...) {
            // No thread could be started (std::system_error, or std::bad_alloc while allocating
            // the thread state): run this range on the calling thread.
            fn(begin, end);
        }
    }
    fn(size_t{0}, std::min(nrows, chunk));
}

// Everything the weight of a bin depends on, besides the bin's own frequency and wavenumber.
struct WeightKey {
    size_t ntr_pad;
    size_t ns_pad;
    double dt;
    double dx;
    double v_min;
    double v_max;
    double taper;
    FKMode mode;
    bool preserve_dc;
};

WeightKey make_weight_key(const FKMuteParams& p, size_t ntr_pad, size_t ns_pad) {
    WeightKey key{};
    key.ntr_pad = ntr_pad;
    key.ns_pad = ns_pad;
    key.dt = p.dt;
    key.dx = p.dx;
    key.v_min = p.v_min;
    key.v_max = p.v_max;
    key.taper = p.taper;
    key.mode = p.mode;
    key.preserve_dc = p.preserve_dc;
    return key;
}

// Exact comparison: a table is reused only when every key field is bit-for-bit the same value.
bool same_key(const WeightKey& a, const WeightKey& b) {
    return a.ntr_pad == b.ntr_pad && a.ns_pad == b.ns_pad && a.dt == b.dt && a.dx == b.dx
        && a.v_min == b.v_min && a.v_max == b.v_max && a.taper == b.taper
        && a.mode == b.mode && a.preserve_dc == b.preserve_dc;
}

struct WeightCacheEntry {
    bool valid = false;
    WeightKey key{};
    std::vector<float> table;  // ntr_pad x nfreq, row i holds the bins of wavenumber index i
};

// Returns the ntr_pad x nfreq float weight table for `params`, building it on a miss. Tables
// are cached per calling thread, at most kWeightCacheEntries of them, and the entry that was
// replaced longest ago is overwritten first. The reference stays valid until the calling thread
// next calls this function.
const std::vector<float>& weight_table(const FKMuteParams& params, size_t ntr_pad, size_t ns_pad,
                                       double df, double dk, size_t nthreads) {
    thread_local std::vector<WeightCacheEntry> cache;
    thread_local size_t next_slot = 0;

    const WeightKey key = make_weight_key(params, ntr_pad, ns_pad);
    for (const WeightCacheEntry& entry : cache) {
        if (entry.valid && same_key(entry.key, key)) {
            return entry.table;
        }
    }

    size_t slot = 0;
    if (cache.size() < kWeightCacheEntries) {
        cache.reserve(kWeightCacheEntries);
        cache.emplace_back();
        slot = cache.size() - 1;
    } else {
        slot = next_slot;
        next_slot = (next_slot + 1) % kWeightCacheEntries;
    }
    WeightCacheEntry& entry = cache[slot];
    entry.valid = false;

    // f_j = j * df for j = 0 .. nfreq - 1; k_i = fftfreq(ntr_pad, dx)[i] (negative above the
    // middle, numpy convention). Both are evaluated exactly as the per-bin loop always did.
    const size_t nfreq = ns_pad / 2 + 1;
    entry.table.resize(ntr_pad * nfreq);
    float* table = entry.table.data();
    const BandCoeffs band = make_band(params);
    const size_t k_pos_max = (ntr_pad - 1) / 2;
    for_each_row_range(ntr_pad, nthreads, ntr_pad * nfreq, [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            const double k_index = (i <= k_pos_max)
                ? static_cast<double>(i)
                : static_cast<double>(i) - static_cast<double>(ntr_pad);
            const double k_cpm = k_index * dk;
            float* row = table + i * nfreq;
            for (size_t j = 0; j < nfreq; ++j) {
                const double f_hz = static_cast<double>(j) * df;
                row[j] = static_cast<float>(bin_weight(f_hz, k_cpm, band));
            }
        }
    });

    entry.key = key;
    entry.valid = true;
    return entry.table;
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
    // Frequency and wavenumber steps. A tiny dt or dx makes 1 / (n * step) overflow to inf, and
    // the zero bin would then be 0 * inf = NaN. Reject that before any buffer is allocated.
    const double df = 1.0 / (static_cast<double>(ns_pad) * params.dt);
    const double dk = 1.0 / (static_cast<double>(ntr_pad) * params.dx);
    if (!(std::isfinite(df) && std::isfinite(dk))) {
        throw std::invalid_argument("fk: dt or dx is too small; frequency or wavenumber step overflows");
    }
    const size_t nthreads = resolve_threads(params.threads);
    const size_t grid_count = ntr_pad * ns_pad;
    const size_t spec_count = ntr_pad * nfreq;

    // Work buffers persist per calling thread, so repeated calls do not allocate or zero the
    // whole grid. Every element that is read is written first (see the padding below).
    // Retention: these buffers and the weight tables (see weight_table) are kept until the
    // calling thread exits, not released after each call. For a 2048 x 2048 gather (padded
    // 3072 x 3072) that is about 113 MB per calling thread: about 75 MB of work buffers plus two
    // 19 MB weight tables.
    thread_local std::vector<float> real;
    thread_local std::vector<std::complex<float>> spec;
    real.resize(grid_count);
    spec.resize(spec_count);

    // Step 3: copy the input into the top-left of the padded buffer, with edge tapers. Only the
    // padding is zeroed: columns nsamples .. ns_pad - 1 of the data rows and every row
    // ntraces .. ntr_pad - 1.
    const std::vector<double> wt = make_edge_taper(ntraces, params.edge_taper_traces);
    const std::vector<double> ws = make_edge_taper(nsamples, params.edge_taper_samples);
    float* real_ptr = real.data();
    for_each_row_range(ntr_pad, nthreads, grid_count, [&](size_t t0, size_t t1) {
        for (size_t t = t0; t < t1; ++t) {
            float* dst = real_ptr + t * ns_pad;
            if (t >= ntraces) {
                std::fill(dst, dst + ns_pad, 0.0f);
                continue;
            }
            const float* src = in + t * nsamples;
            for (size_t s = 0; s < nsamples; ++s) {
                dst[s] = static_cast<float>(static_cast<double>(src[s]) * wt[t] * ws[s]);
            }
            std::fill(dst + nsamples, dst + ns_pad, 0.0f);
        }
    });

    // Step 4: forward real-to-complex 2D FFT over {traces, samples}. Strides in bytes.
    const pocketfft::shape_t shape{ntr_pad, ns_pad};
    const pocketfft::shape_t axes{0, 1};
    const pocketfft::stride_t stride_real{
        static_cast<std::ptrdiff_t>(ns_pad * sizeof(float)),
        static_cast<std::ptrdiff_t>(sizeof(float))};
    const pocketfft::stride_t stride_spec{
        static_cast<std::ptrdiff_t>(nfreq * sizeof(std::complex<float>)),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};

    pocketfft::r2c<float>(shape, stride_real, stride_spec, axes, true,
                          real_ptr, spec.data(), 1.0f, nthreads);

    // Step 5: weight every bin by fk_mute_weight(f_j, k_i), from the per-thread weight cache.
    // Nyquist row and column bins are not special-cased: the c2r inverse keeps only their
    // Hermitian part, which matches numpy.fft.irfft2(rfft2(x) * w). Energy there requires
    // aliased data (FK_PIPELINE.md section 3).
    const std::vector<float>& weights = weight_table(params, ntr_pad, ns_pad, df, dk, nthreads);
    const float* weight_ptr = weights.data();
    std::complex<float>* spec_ptr = spec.data();
    for_each_row_range(ntr_pad, nthreads, spec_count, [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            // complex<float> is layout-compatible with float[2] ([complex.numbers]), so the real
            // and imaginary parts are scaled by the same real weight.
            float* p = reinterpret_cast<float*>(spec_ptr + i * nfreq);
            const float* w = weight_ptr + i * nfreq;
            for (size_t j = 0; j < nfreq; ++j) {
                p[2 * j] *= w[j];
                p[2 * j + 1] *= w[j];
            }
        }
    });

    // Step 6: complex-to-real inverse 2D FFT with scale 1 / (ntr_pad * ns_pad).
    // The padded buffer is free once the forward transform is done; reuse it for the result.
    const float inv_scale = static_cast<float>(
        1.0 / (static_cast<double>(ntr_pad) * static_cast<double>(ns_pad)));
    pocketfft::c2r<float>(shape, stride_spec, stride_real, axes, false,
                          spec_ptr, real_ptr, inv_scale, nthreads);

    // Step 7: copy the valid top-left block out. `in` is no longer read, so aliasing is safe.
    for_each_row_range(ntraces, nthreads, ntraces * nsamples, [&](size_t t0, size_t t1) {
        for (size_t t = t0; t < t1; ++t) {
            const float* src = real_ptr + t * ns_pad;
            float* dst = out + t * nsamples;
            std::copy(src, src + nsamples, dst);
        }
    });
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
