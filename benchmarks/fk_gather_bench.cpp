// Benchmark for basalt::fk_filter_gather (the end-to-end f-k pipeline, include/basalt/fk.hpp).
//
// Each sample is one full call on a synthetic trace-major gather: edge tapers, zero-padding,
// real-to-complex 2D FFT, velocity-band weighting, complex-to-real inverse and copy-out. The
// call is timed with steady_clock; reported are the median and 90th percentile of the per-call
// time and gathers per second derived from the median.
//
// The legacy power-of-two path (fft2d_forward, apply_fk_filter, fft2d_inverse, as in the README
// "Usage" section) is timed for comparison, labelled "legacy-c2c", on sizes whose two dimensions
// are both powers of two. It is single-threaded, uses a full complex transform and no padding or
// edge tapers, so it does different work and its output is not compared with the pipeline's.
//
// The "pocketfft-transform-only" rows time only the pipeline's own r2c and c2r calls on the same
// padded grid, with the same pocketfft version. They are the floor for the transform inside
// basalt-c++, so the difference between the two rows is the cost of the work around it.
//
// Usage: fk_gather_bench [--sizes NTRxNS,...] [--threads N,...] [--iterations N] [--warmup N]
//                        [--pad P] [--csv-out FILE]
// Run `fk_gather_bench --help` for details. Contract: docs/design/FK_PIPELINE.md.

#include "basalt/arena.hpp"
#include "basalt/filter/fk_filter.hpp"
#include "basalt/fk.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/fft2d.hpp"

// The same header fk_pipeline.cpp uses; the benchmark needs it for the transform-only floor.
#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct GatherSize {
    size_t ntraces;
    size_t nsamples;
};

struct Config {
    std::vector<GatherSize> sizes{{240, 1001}, {1001, 1500}, {1024, 1024}, {2048, 2048}};
    std::vector<size_t> threads;  // empty means {1, hardware concurrency}
    size_t iterations = 21;
    size_t warmup = 3;
    double pad_factor = 1.5;
    std::string csv_out;
};

struct TimingStats {
    double median_ms = 0.0;
    double p90_ms = 0.0;
    double min_ms = 0.0;
    double mean_ms = 0.0;
};

struct Row {
    std::string impl;
    GatherSize size{0, 0};
    size_t ntr_pad = 0;
    size_t ns_pad = 0;
    size_t threads = 1;
    TimingStats stats;
    double checksum = 0.0;  // sum of the output samples; a sanity figure, not a correctness check
};

void print_usage(std::ostream& os) {
    os << "usage: fk_gather_bench [options]\n"
       << "  --sizes LIST       comma-separated NTRxNS gathers (default 240x1001,1001x1500,\n"
       << "                     1024x1024,2048x2048)\n"
       << "  --threads LIST     comma-separated FFT thread counts; 0 = hardware concurrency\n"
       << "                     (default 1,hardware concurrency)\n"
       << "  --iterations N     timed calls per configuration (default 21)\n"
       << "  --warmup N         untimed calls per configuration (default 3)\n"
       << "  --pad P            pad_factor, >= 1 (default 1.5)\n"
       << "  --csv-out FILE     also write the results as CSV\n";
}

[[noreturn]] void usage_error(const std::string& msg) {
    std::cerr << "fk_gather_bench: " << msg << "\n";
    print_usage(std::cerr);
    std::exit(2);
}

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : text) {
        if (c == sep) {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    parts.push_back(current);
    return parts;
}

// Parses an unsigned decimal integer that must consume the whole token and be >= min_value.
size_t parse_count(const std::string& text, const char* what, size_t min_value) {
    if (text.empty() || !std::isdigit(static_cast<unsigned char>(text[0]))) {
        usage_error(std::string("invalid ") + what + " '" + text + "'");
    }
    size_t pos = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &pos, 10);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos != text.size() || value < min_value) {
        usage_error(std::string("invalid ") + what + " '" + text + "'");
    }
    return static_cast<size_t>(value);
}

double parse_pad(const std::string& text) {
    size_t pos = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &pos);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos != text.size() || !std::isfinite(value) || value < 1.0) {
        usage_error("invalid pad factor '" + text + "' (need a number >= 1)");
    }
    return value;
}

std::vector<GatherSize> parse_sizes(const std::string& text) {
    std::vector<GatherSize> sizes;
    for (const std::string& item : split(text, ',')) {
        const size_t x = item.find_first_of("xX");
        if (x == std::string::npos) {
            usage_error("size must be NTRxNS, got '" + item + "'");
        }
        const size_t ntraces = parse_count(item.substr(0, x), "ntraces", 1);
        const size_t nsamples = parse_count(item.substr(x + 1), "nsamples", 1);
        sizes.push_back(GatherSize{ntraces, nsamples});
    }
    return sizes;
}

std::vector<size_t> parse_threads(const std::string& text) {
    std::vector<size_t> threads;
    for (const std::string& item : split(text, ',')) {
        threads.push_back(parse_count(item, "thread count", 0));
    }
    return threads;
}

size_t hardware_threads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<size_t>(hc);
}

// Resolves 0 to the hardware concurrency and drops duplicates, keeping the requested order.
std::vector<size_t> resolve_thread_list(const std::vector<size_t>& requested) {
    const std::vector<size_t> base = requested.empty()
        ? std::vector<size_t>{1, hardware_threads()}
        : requested;
    std::vector<size_t> resolved;
    for (size_t t : base) {
        const size_t value = (t == 0) ? hardware_threads() : t;
        if (std::find(resolved.begin(), resolved.end(), value) == resolved.end()) {
            resolved.push_back(value);
        }
    }
    return resolved;
}

bool is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

// Deterministic uniform noise. The FFT cost does not depend on the sample values, and a fixed
// seed keeps every configuration on the same input.
std::vector<float> make_gather(const GatherSize& s) {
    std::mt19937 rng(20260101u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> gather(s.ntraces * s.nsamples);
    for (float& v : gather) {
        v = dist(rng);
    }
    return gather;
}

double checksum(const float* data, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return sum;
}

TimingStats summarize(std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    const size_t n = ms.size();
    TimingStats stats;
    stats.median_ms = (n % 2 == 1) ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
    // Nearest-rank 90th percentile: rank = ceil(0.9 * n), counted from 1.
    const size_t rank = std::max<size_t>((9 * n + 9) / 10, 1);
    stats.p90_ms = ms[std::min(rank, n) - 1];
    stats.min_ms = ms.front();
    stats.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(n);
    return stats;
}

// Runs `call` `warmup` times untimed, then `iterations` times with each call timed on its own.
template <typename Fn>
std::vector<double> time_calls(Fn&& call, size_t warmup, size_t iterations) {
    for (size_t i = 0; i < warmup; ++i) {
        call();
    }
    std::vector<double> ms;
    ms.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        const Clock::time_point t0 = Clock::now();
        call();
        const Clock::time_point t1 = Clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return ms;
}

Row run_pipeline(const GatherSize& s, size_t threads, const Config& cfg,
                 const std::vector<float>& in, std::vector<float>& out) {
    basalt::FKMuteParams params;  // library defaults except the pad factor and thread count
    params.pad_factor = cfg.pad_factor;
    params.threads = threads;

    auto call = [&]() {
        basalt::fk_filter_gather(in.data(), out.data(), s.ntraces, s.nsamples, params);
    };
    const std::vector<double> ms = time_calls(call, cfg.warmup, cfg.iterations);

    Row row;
    row.impl = "basalt-c++";
    row.size = s;
    row.ntr_pad = basalt::fk_padded_traces(s.ntraces, params);
    row.ns_pad = basalt::fk_padded_samples(s.nsamples, params);
    row.threads = threads;
    row.stats = summarize(ms);
    row.checksum = checksum(out.data(), out.size());
    return row;
}

// Legacy power-of-two path. Rows are the frequency (time) axis and columns the wavenumber
// (trace) axis, as documented for apply_fk_filter, so the trace-major gather is transposed once
// outside the timed region. Each timed call copies the gather into the SoA arrays first, as the
// pipeline copies its input, and no output copy is made.
Row run_legacy(const GatherSize& s, const Config& cfg, const std::vector<float>& in) {
    const size_t rows = s.nsamples;
    const size_t cols = s.ntraces;
    const size_t total = rows * cols;

    std::vector<float> src(total);
    for (size_t t = 0; t < s.ntraces; ++t) {
        for (size_t k = 0; k < s.nsamples; ++k) {
            src[k * cols + t] = in[t * s.nsamples + k];
        }
    }

    basalt::MemoryArena data_arena(total * sizeof(float) * 8 + 65536);
    basalt::kernel::ComplexSoA data(data_arena, total);
    basalt::MemoryArena scratch(total * sizeof(float) * 4 + 65536);
    const basalt::filter::FKParams params;  // legacy defaults: dt 4 ms, dx 25 m, band +-1500 m/s

    auto call = [&]() {
        std::copy(src.begin(), src.end(), data.real);
        std::fill_n(data.imag, total, 0.0f);
        scratch.reset();
        basalt::kernel::fft2d_forward(data.real, data.imag, rows, cols, scratch);
        basalt::filter::apply_fk_filter(data, rows, cols, params);
        scratch.reset();
        basalt::kernel::fft2d_inverse(data.real, data.imag, rows, cols, scratch);
    };
    const std::vector<double> ms = time_calls(call, cfg.warmup, cfg.iterations);

    Row row;
    row.impl = "legacy-c2c";
    row.size = s;
    row.ntr_pad = s.ntraces;  // the legacy path does not pad
    row.ns_pad = s.nsamples;
    row.threads = 1;
    row.stats = summarize(ms);
    row.checksum = checksum(data.real, total);
    return row;
}

// Floor for the pipeline's transform: the same padded real grid and the same pocketfft r2c and
// c2r calls that fk_filter_gather makes, with no taper, weight or copy-in work. Buffers are
// allocated and filled once, outside the timed interval.
Row run_transform_floor(const GatherSize& s, size_t threads, const Config& cfg,
                        const std::vector<float>& in) {
    basalt::FKMuteParams params;
    params.pad_factor = cfg.pad_factor;
    const size_t ntr_pad = basalt::fk_padded_traces(s.ntraces, params);
    const size_t ns_pad = basalt::fk_padded_samples(s.nsamples, params);
    const size_t nfreq = ns_pad / 2 + 1;

    std::vector<float> real(ntr_pad * ns_pad, 0.0f);
    for (size_t t = 0; t < s.ntraces; ++t) {
        std::copy_n(in.data() + t * s.nsamples, s.nsamples, real.data() + t * ns_pad);
    }
    std::vector<std::complex<float>> spec(ntr_pad * nfreq);
    std::vector<float> back(ntr_pad * ns_pad);

    const pocketfft::shape_t shape{ntr_pad, ns_pad};
    const pocketfft::shape_t axes{0, 1};
    const pocketfft::stride_t stride_real{
        static_cast<std::ptrdiff_t>(ns_pad * sizeof(float)),
        static_cast<std::ptrdiff_t>(sizeof(float))};
    const pocketfft::stride_t stride_spec{
        static_cast<std::ptrdiff_t>(nfreq * sizeof(std::complex<float>)),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
    const float inv_scale = static_cast<float>(
        1.0 / (static_cast<double>(ntr_pad) * static_cast<double>(ns_pad)));

    auto call = [&]() {
        pocketfft::r2c<float>(shape, stride_real, stride_spec, axes, true,
                              real.data(), spec.data(), 1.0f, threads);
        pocketfft::c2r<float>(shape, stride_spec, stride_real, axes, false,
                              spec.data(), back.data(), inv_scale, threads);
    };
    const std::vector<double> ms = time_calls(call, cfg.warmup, cfg.iterations);

    Row row;
    row.impl = "pocketfft-transform-only";
    row.size = s;
    row.ntr_pad = ntr_pad;
    row.ns_pad = ns_pad;
    row.threads = threads;
    row.stats = summarize(ms);
    row.checksum = checksum(back.data(), back.size());
    return row;
}

void print_table(const std::vector<Row>& rows, const Config& cfg) {
    std::cout << std::left << std::setw(12) << "impl" << std::right
              << std::setw(9) << "ntraces" << std::setw(9) << "nsamps"
              << std::setw(9) << "ntr_pad" << std::setw(9) << "ns_pad"
              << std::setw(9) << "threads" << std::setw(12) << "median_ms"
              << std::setw(10) << "p90_ms" << std::setw(10) << "min_ms"
              << std::setw(10) << "mean_ms" << std::setw(13) << "gathers/s" << "\n";
    std::cout << std::fixed << std::setprecision(3);
    for (const Row& r : rows) {
        const double gathers_per_s = 1000.0 / r.stats.median_ms;
        std::cout << std::left << std::setw(12) << r.impl << std::right
                  << std::setw(9) << r.size.ntraces << std::setw(9) << r.size.nsamples
                  << std::setw(9) << r.ntr_pad << std::setw(9) << r.ns_pad
                  << std::setw(9) << r.threads << std::setw(12) << r.stats.median_ms
                  << std::setw(10) << r.stats.p90_ms << std::setw(10) << r.stats.min_ms
                  << std::setw(10) << r.stats.mean_ms << std::setw(13)
                  << std::setprecision(2) << gathers_per_s << std::setprecision(3) << "\n";
    }
    std::cout << "iterations=" << cfg.iterations << " warmup=" << cfg.warmup
              << " pad_factor=" << cfg.pad_factor << "\n";
}

void write_csv(const std::string& path, const std::vector<Row>& rows, const Config& cfg) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open CSV output '" + path + "'");
    }
    out << "impl,ntraces,nsamples,ntraces_padded,nsamples_padded,threads,iterations,warmup,"
        << "pad_factor,median_ms,p90_ms,min_ms,mean_ms,gathers_per_s,checksum\n";
    out << std::setprecision(9);
    for (const Row& r : rows) {
        out << r.impl << ',' << r.size.ntraces << ',' << r.size.nsamples << ','
            << r.ntr_pad << ',' << r.ns_pad << ',' << r.threads << ','
            << cfg.iterations << ',' << cfg.warmup << ',' << cfg.pad_factor << ','
            << r.stats.median_ms << ',' << r.stats.p90_ms << ',' << r.stats.min_ms << ','
            << r.stats.mean_ms << ',' << 1000.0 / r.stats.median_ms << ','
            << r.checksum << '\n';
    }
    if (!out) {
        throw std::runtime_error("write failed for CSV output '" + path + "'");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) {
                    usage_error("missing value for " + arg);
                }
                ++i;
                return argv[i];
            };
            if (arg == "--sizes") {
                cfg.sizes = parse_sizes(value());
            } else if (arg == "--threads") {
                cfg.threads = parse_threads(value());
            } else if (arg == "--iterations") {
                cfg.iterations = parse_count(value(), "iterations", 1);
            } else if (arg == "--warmup") {
                cfg.warmup = parse_count(value(), "warmup", 0);
            } else if (arg == "--pad") {
                cfg.pad_factor = parse_pad(value());
            } else if (arg == "--csv-out") {
                cfg.csv_out = value();
            } else if (arg == "--help" || arg == "-h") {
                print_usage(std::cout);
                return 0;
            } else {
                usage_error("unknown argument '" + arg + "'");
            }
        }
        if (cfg.sizes.empty()) {
            usage_error("--sizes needs at least one gather size");
        }

#ifdef NDEBUG
        const char* build = "optimized (NDEBUG)";
#else
        const char* build = "debug (assertions enabled)";
#endif
        std::cout << "fk_gather_bench: " << build << ", compiler " << __VERSION__ << "\n";
        std::cout << "hardware threads: " << hardware_threads() << "\n";

        const std::vector<size_t> threads = resolve_thread_list(cfg.threads);
        std::vector<Row> results;
        for (const GatherSize& s : cfg.sizes) {
            const std::vector<float> in = make_gather(s);
            std::vector<float> out(in.size());
            for (size_t t : threads) {
                results.push_back(run_pipeline(s, t, cfg, in, out));
                results.push_back(run_transform_floor(s, t, cfg, in));
            }
            if (is_power_of_two(s.ntraces) && is_power_of_two(s.nsamples)) {
                results.push_back(run_legacy(s, cfg, in));
            } else {
                std::cerr << "note: legacy-c2c skipped for " << s.ntraces << "x" << s.nsamples
                          << " (needs power-of-two sizes)\n";
            }
        }

        print_table(results, cfg);
        if (!cfg.csv_out.empty()) {
            write_csv(cfg.csv_out, results, cfg);
            std::cout << "csv: " << cfg.csv_out << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fk_gather_bench: " << e.what() << "\n";
        return 1;
    }
}
