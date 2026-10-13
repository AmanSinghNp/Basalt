// Benchmark for the SEG-Y reader and writer (basalt/io/segy_file.hpp).
//
// Writes a synthetic file of about --size-mb megabytes in IBM32 and in IEEE32, then times
// read_segy and write_segy with 1 and 4 threads (--threads) and reports the throughput in MiB/s
// (median of --runs repetitions, with min and max). Setup work (generating samples, creating the
// input files, deleting the output between runs) is not timed. Synthetic files are written under
// --dir and deleted at the end, including after an error.
//
// The threads=1 rows include file I/O overlap: even with one thread the library reads or writes the
// next chunk on a helper thread while the current one is decoded or encoded.
//
// Build with -DBASALT_SEGY_BENCH_NO_THREADS to compile against a library whose read_segy and
// write_segy take no thread count (the pre-threads baseline, commit 5009e1e). Those rows are then
// reported as n/a, since the library is single-threaded there. The baseline signatures mangle to
// different names, so the baseline header must be first on the include path or the link fails:
//   git show 5009e1e:include/basalt/io/segy_file.hpp > DIR/basalt/io/segy_file.hpp
//   c++ -std=c++17 -O2 -DBASALT_SEGY_BENCH_NO_THREADS -IDIR -I<repo>/include -c benchmarks/segy_io_bench.cpp
//   c++ segy_io_bench.o <baseline-build>/libbasalt_core.a -o segy_io_bench
//
// Usage: segy_io_bench [--size-mb N] [--dir PATH] [--runs N] [--nsamples N] [--threads a,b,...]
// Run `segy_io_bench --help` for details.

#include "basalt/io/segy_file.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using basalt::io::SampleFormat;
using basalt::io::SegyData;

#if defined(BASALT_SEGY_BENCH_NO_THREADS)
constexpr bool kThreadsSupported = false;
#else
constexpr bool kThreadsSupported = true;
#endif

constexpr size_t kTraceHeaderBytes = 240;
constexpr size_t kTextAndBinaryHeaderBytes = 3600;

struct Config {
    size_t size_mb = 512;
    std::string dir = "/tmp";
    size_t runs = 3;
    size_t nsamples = 1000;
    std::vector<size_t> threads{1, 4};
};

void print_usage(std::ostream& os) {
    os << "Usage: segy_io_bench [--size-mb N] [--dir PATH] [--runs N] [--nsamples N] [--threads a,b,...]\n"
       << "\n"
       << "  --size-mb N     approximate size of each synthetic file in MiB (default 512)\n"
       << "  --dir PATH      directory for the synthetic files (default /tmp); deleted at the end\n"
       << "  --runs N        timed repetitions per configuration; the median is reported (default 3)\n"
       << "  --nsamples N    samples per trace, 1..65535 (default 1000)\n"
       << "  --threads LIST  thread counts to time, comma separated; 0 = hardware concurrency\n"
       << "                  (default 1,4)\n"
       << "  -h, --help      show this help\n";
}

std::vector<size_t> parse_list(const std::string& text) {
    std::vector<size_t> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const std::string item = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            throw std::invalid_argument("empty entry in list '" + text + "'");
        }
        out.push_back(static_cast<size_t>(std::stoull(item)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(std::cout);
            std::exit(0);
        } else if (arg == "--size-mb") {
            cfg.size_mb = static_cast<size_t>(std::stoull(value()));
        } else if (arg == "--dir") {
            cfg.dir = value();
        } else if (arg == "--runs") {
            cfg.runs = static_cast<size_t>(std::stoull(value()));
        } else if (arg == "--nsamples") {
            cfg.nsamples = static_cast<size_t>(std::stoull(value()));
        } else if (arg == "--threads") {
            cfg.threads = parse_list(value());
        } else {
            throw std::invalid_argument("unknown argument '" + arg + "'");
        }
    }
    if (cfg.size_mb == 0 || cfg.runs == 0 || cfg.nsamples == 0 || cfg.nsamples > 65535 || cfg.threads.empty()) {
        throw std::invalid_argument("--size-mb, --runs and --nsamples must be positive; --nsamples <= 65535");
    }
    return cfg;
}

// Thread-count dispatch: the threaded API when the library has it.
SegyData read_file(const std::string& path, size_t threads) {
#if defined(BASALT_SEGY_BENCH_NO_THREADS)
    (void)threads;
    return basalt::io::read_segy(path);
#else
    return basalt::io::read_segy(path, threads);
#endif
}

void write_file(const std::string& path, const SegyData& data, SampleFormat fmt, size_t threads) {
#if defined(BASALT_SEGY_BENCH_NO_THREADS)
    (void)threads;
    basalt::io::write_segy(path, data, fmt);
#else
    basalt::io::write_segy(path, data, fmt, threads);
#endif
}

// Removes the listed files when it goes out of scope, so an error does not leave 512 MB behind.
class FileCleanup {
public:
    ~FileCleanup() {
        for (const std::string& p : paths_) {
            std::remove(p.c_str());
        }
    }
    void add(const std::string& p) { paths_.push_back(p); }

private:
    std::vector<std::string> paths_;
};

struct Stats {
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
};

Stats summarise(std::vector<double> mibps) {
    std::sort(mibps.begin(), mibps.end());
    Stats s;
    s.min = mibps.front();
    s.max = mibps.back();
    const size_t n = mibps.size();
    s.median = (n % 2 == 1) ? mibps[n / 2] : 0.5 * (mibps[n / 2 - 1] + mibps[n / 2]);
    return s;
}

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string format_name(SampleFormat fmt) {
    return fmt == SampleFormat::IBM32 ? "IBM32" : "IEEE32";
}

void print_row(const std::string& op, SampleFormat fmt, size_t threads, const Stats& s) {
    std::cout << std::left << std::setw(7) << op << std::setw(8) << format_name(fmt) << std::setw(9)
              << threads << std::right << std::fixed << std::setprecision(1) << std::setw(11) << s.median
              << std::setw(11) << s.min << std::setw(11) << s.max << "\n";
}

void print_unsupported(const std::string& op, SampleFormat fmt, size_t threads) {
    std::cout << std::left << std::setw(7) << op << std::setw(8) << format_name(fmt) << std::setw(9)
              << threads << "n/a (library has no thread count; single-threaded)\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);

        // Samples per trace fix the trace length; the trace count fills about size_mb.
        const size_t trace_bytes = kTraceHeaderBytes + cfg.nsamples * 4;
        const size_t target_bytes = cfg.size_mb * (size_t{1} << 20);
        const size_t ntraces = std::max<size_t>(1, (target_bytes - kTextAndBinaryHeaderBytes) / trace_bytes);
        const size_t total_samples = ntraces * cfg.nsamples;
        const double file_mib =
            static_cast<double>(kTextAndBinaryHeaderBytes + ntraces * trace_bytes) / (1 << 20);

        std::cout << "SEG-Y I/O benchmark: " << ntraces << " traces x " << cfg.nsamples << " samples, "
                  << std::fixed << std::setprecision(1) << file_mib << " MiB per file, " << cfg.runs
                  << " run(s) per configuration\n";

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
        std::vector<float> samples(total_samples);
        for (float& x : samples) {
            x = dist(rng);
        }
        const SegyData master = basalt::io::make_segy(samples, ntraces, cfg.nsamples, 0.002, SampleFormat::IEEE32);
        samples.clear();
        samples.shrink_to_fit();

        const std::string base = cfg.dir + "/basalt_segy_bench";
        FileCleanup cleanup;
        const std::string in_path[2] = {base + "_in_ibm.segy", base + "_in_ieee.segy"};
        const std::string out_path = base + "_out.segy";
        const SampleFormat formats[2] = {SampleFormat::IBM32, SampleFormat::IEEE32};
        for (const std::string& p : in_path) {
            cleanup.add(p);
        }
        cleanup.add(out_path);
        basalt::io::write_segy(in_path[0], master, SampleFormat::IBM32);
        basalt::io::write_segy(in_path[1], master, SampleFormat::IEEE32);
        sync();  // setup writes are flushed before any timed run starts

        std::cout << std::left << std::setw(7) << "op" << std::setw(8) << "format" << std::setw(9) << "threads"
                  << std::right << std::setw(11) << "MiB/s" << std::setw(11) << "min" << std::setw(11) << "max"
                  << "\n";

        for (size_t f = 0; f < 2; ++f) {
            const SampleFormat fmt = formats[f];
            for (size_t threads : cfg.threads) {
                if (!kThreadsSupported && threads != 1) {
                    print_unsupported("read", fmt, threads);
                    continue;
                }
                std::vector<double> mibps;
                for (size_t r = 0; r < cfg.runs; ++r) {
                    const auto t0 = Clock::now();
                    const SegyData back = read_file(in_path[f], threads);
                    const double s = seconds_since(t0);
                    if (back.ntraces != ntraces) {
                        throw std::runtime_error("read back " + std::to_string(back.ntraces) + " traces, expected " +
                                                 std::to_string(ntraces));
                    }
                    mibps.push_back(file_mib / s);  // destroying the samples is after the timer stops
                }
                print_row("read", fmt, threads, summarise(mibps));
            }
            // Untimed warm-up write: the first write of a new file pays for block allocation on the
            // file system, which would otherwise be charged to one timed run only.
            write_file(out_path, master, fmt, 1);
            std::remove(out_path.c_str());
            sync();
            for (size_t threads : cfg.threads) {
                if (!kThreadsSupported && threads != 1) {
                    print_unsupported("write", fmt, threads);
                    continue;
                }
                std::vector<double> mibps;
                for (size_t r = 0; r < cfg.runs; ++r) {
                    // Untimed: remove the previous output and flush dirty pages, so the writeback of one
                    // run is not charged to the next one.
                    std::remove(out_path.c_str());
                    sync();
                    const auto t0 = Clock::now();
                    write_file(out_path, master, fmt, threads);
                    mibps.push_back(file_mib / seconds_since(t0));
                }
                print_row("write", fmt, threads, summarise(mibps));
            }
        }
        std::cout << "(MiB/s = file bytes / wall-clock seconds; 1 MiB = 2^20 bytes)\n";
        return 0;
    } catch (const std::invalid_argument& e) {
        std::cerr << "segy_io_bench: " << e.what() << "\n";
        print_usage(std::cerr);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "segy_io_bench: " << e.what() << "\n";
        return 1;
    }
}
