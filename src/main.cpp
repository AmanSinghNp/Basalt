// basalt command-line tool: inspects SEG-Y files and runs the f-k velocity-band mute on them.
// Command and option contract: docs/design/FK_PIPELINE.md, section 6.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "basalt/fk.hpp"
#include "basalt/io/segy_file.hpp"
#include "basalt/kernel/cpu_state.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitRuntime = 1;
constexpr int kExitUsage = 2;

// Byte offsets (0-based) of the sample interval fields in the headers.
constexpr size_t kBinSampleIntervalOff = 16; // binary header bytes 3217-3218
constexpr size_t kTrSampleIntervalOff = 116; // trace header bytes 117-118
constexpr size_t kTextCardWidth = 80;        // SEG-Y text header card width

// Raised for malformed command lines. main reports it with the usage text and exit code 2.
class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void print_usage(std::ostream& os) {
    os << "usage:\n"
       << "  basalt info   <in.segy> [--verbose]\n"
       << "  basalt filter <in.segy> <out.segy> [--vmin V] [--vmax V] [--taper T]"
       << " [--mode reject|pass]\n"
       << "                                     [--dx M] [--dt S] [--pad P] [--edge-taper N]\n"
       << "                                     [--threads N] [--gather-size N]"
       << " [--format ibm|ieee]\n"
       << "                                     [--verbose]\n"
       << "  basalt --version | --help\n";
}

void print_help(std::ostream& os) {
    print_usage(os);
    os << "\nfilter options (defaults):\n"
       << "  --vmin V         lower edge of the velocity band, m/s (-1500)\n"
       << "  --vmax V         upper edge of the velocity band, m/s (1500)\n"
       << "  --taper T        fractional edge width, 0 <= T < 1 (0.1)\n"
       << "  --mode M         reject (mute the band) or pass (keep only the band) (reject)\n"
       << "  --dx M           trace spacing, m (25)\n"
       << "  --dt S           sample interval, s, a whole number of microseconds in\n"
       << "                   [1e-6, 0.065535] (from the file header)\n"
       << "  --pad P          FFT padding factor, >= 1 (1.5)\n"
       << "  --edge-taper N   cosine taper over the first and last N traces of each block (4).\n"
       << "                   In a block shorter than 2N traces the two tapers overlap and\n"
       << "                   their weights multiply; a warning is printed\n"
       << "  --threads N      worker threads, 0 = all cores (1): used to read and write the\n"
       << "                   file and for the FFTs of a single gather. When --gather-size\n"
       << "                   gives two or more gathers, they run in parallel, each on one\n"
       << "                   thread, and the output is identical for every N\n"
       << "  --gather-size N  filter consecutive blocks of N traces (whole file). The blocks\n"
       << "                   run in parallel; a short last block runs alongside them\n"
       << "  --format F       output sample format, ibm or ieee (input format if\n"
       << "                   IBM32 or IEEE32, otherwise ieee)\n"
       << "\nother options:\n"
       << "  --verbose        print SIMD and FTZ/DAZ diagnostics to stderr\n"
       << "  --help, -h       show this help\n"
       << "  --version        print the version\n"
       << "\nexit status: 0 success, 1 runtime error, 2 usage error\n";
}

// Parses a finite decimal number. The whole token must be consumed. Only decimal characters
// are accepted: strtod alone would also take hex floats (0x1p3) and the words inf and nan.
double parse_number(const std::string& flag, const std::string& text) {
    const auto bad = [&]() { return UsageError("invalid number '" + text + "' for " + flag); };
    if (text.empty() || text.find_first_not_of("0123456789+-.eE") != std::string::npos) {
        throw bad();
    }
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size() || !std::isfinite(value)) {
        throw bad();
    }
    // strtod sets ERANGE when the value underflows to zero or to a subnormal. Such values are
    // rejected by name: the flush-to-zero mode set in main would turn them into zero silently.
    if (errno == ERANGE && std::fabs(value) < std::numeric_limits<double>::min()) {
        throw UsageError("number '" + text + "' is too small for " + flag);
    }
    return value;
}

// Parses a non-negative integer count. Values that are not whole numbers are rejected.
size_t parse_count(const std::string& flag, const std::string& text) {
    constexpr double kMaxCount = 9007199254740992.0; // 2^53, far above any real trace count
    const double value = parse_number(flag, text);
    if (value < 0.0 || value != std::floor(value) || value > kMaxCount) {
        throw UsageError(flag + " expects a non-negative integer, got '" + text + "'");
    }
    return static_cast<size_t>(value);
}

size_t parse_positive_count(const std::string& flag, const std::string& text) {
    const size_t value = parse_count(flag, text);
    if (value == 0) {
        throw UsageError(flag + " must be at least 1");
    }
    return value;
}

basalt::FKMode parse_mode(const std::string& text) {
    if (text == "reject") {
        return basalt::FKMode::Reject;
    }
    if (text == "pass") {
        return basalt::FKMode::Pass;
    }
    throw UsageError("--mode expects 'reject' or 'pass', got '" + text + "'");
}

basalt::io::SampleFormat parse_format(const std::string& text) {
    if (text == "ibm") {
        return basalt::io::SampleFormat::IBM32;
    }
    if (text == "ieee") {
        return basalt::io::SampleFormat::IEEE32;
    }
    throw UsageError("--format expects 'ibm' or 'ieee', got '" + text + "'");
}

// SEG-Y stores the sample interval as a 16-bit count of microseconds.
uint16_t dt_to_microseconds(double dt) {
    const double scaled = dt * 1e6;
    const double us = std::round(scaled);
    if (us < 1.0 || us > 65535.0 || std::fabs(scaled - us) > 1e-6) {
        throw UsageError("--dt must be a whole number of microseconds between 1e-6 and 0.065535 s");
    }
    return static_cast<uint16_t>(us);
}

void put_be16(uint8_t* p, uint16_t value) {
    p[0] = static_cast<uint8_t>(value >> 8);
    p[1] = static_cast<uint8_t>(value & 0xFFu);
}

uint16_t get_be16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

// Writes the sample interval into the binary header and every trace header that already
// declares one, so that the output file records the interval the filter actually used.
void set_sample_interval(basalt::io::SegyData& data, uint16_t interval_us) {
    put_be16(&data.raw_binary_header[kBinSampleIntervalOff], interval_us);
    for (auto& header : data.raw_trace_headers) {
        if (get_be16(&header[kTrSampleIntervalOff]) != 0) {
            put_be16(&header[kTrSampleIntervalOff], interval_us);
        }
    }
}

const char* format_name(basalt::io::SampleFormat format) {
    switch (format) {
    case basalt::io::SampleFormat::IBM32:
        return "IBM32";
    case basalt::io::SampleFormat::INT32:
        return "INT32";
    case basalt::io::SampleFormat::INT16:
        return "INT16";
    case basalt::io::SampleFormat::IEEE32:
        return "IEEE32";
    case basalt::io::SampleFormat::INT8:
        return "INT8";
    }
    return "unknown";
}

// The thread count a request of `requested` threads stands for; 0 means hardware concurrency.
size_t resolve_threads(size_t requested) {
    if (requested != 0) {
        return requested;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<size_t>(hc);
}

// Filter output keeps the input encoding when the writer supports it, otherwise IEEE32.
basalt::io::SampleFormat default_output_format(basalt::io::SampleFormat input) {
    if (input == basalt::io::SampleFormat::IBM32 || input == basalt::io::SampleFormat::IEEE32) {
        return input;
    }
    return basalt::io::SampleFormat::IEEE32;
}

// The first `count` lines of a text header. Standard headers are 40 cards of 80 characters
// with no line breaks; text that does contain newlines is split on them instead.
// Control characters are shown as spaces and trailing blanks are dropped.
std::vector<std::string> leading_text_lines(const std::string& text, size_t count) {
    const bool has_newlines = text.find('\n') != std::string::npos;
    std::vector<std::string> lines;
    size_t pos = 0;
    while (pos < text.size() && lines.size() < count) {
        size_t end;
        if (has_newlines) {
            end = text.find('\n', pos);
            if (end == std::string::npos) {
                end = text.size();
            }
        } else {
            end = std::min(text.size(), pos + kTextCardWidth);
        }
        std::string line = text.substr(pos, end - pos);
        for (char& c : line) {
            if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) {
                c = ' ';
            }
        }
        while (!line.empty() && line.back() == ' ') {
            line.pop_back();
        }
        lines.push_back(line);
        pos = has_newlines ? end + 1 : end;
    }
    return lines;
}

void print_diagnostics() {
    std::cerr << "basalt " << BASALT_VERSION << " diagnostics\n";
#ifdef __AVX2__
    std::cerr << "  AVX2: compiled in\n";
#else
    std::cerr << "  AVX2: not compiled in\n";
#endif
#ifdef __FMA__
    std::cerr << "  FMA: compiled in\n";
#else
    std::cerr << "  FMA: not compiled in\n";
#endif
    std::cerr << "  FTZ/DAZ: " << (basalt::kernel::check_ftz_daz() ? "enabled" : "disabled")
              << "\n";
}

int run_info(const std::vector<std::string>& args) {
    bool verbose = false;
    std::vector<std::string> positional;
    for (const std::string& arg : args) {
        if (arg == "--help" || arg == "-h") {
            print_help(std::cout);
            return kExitOk;
        }
        if (arg == "--verbose") {
            verbose = true;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw UsageError("unknown option '" + arg + "'");
        }
        positional.push_back(arg);
    }
    if (positional.size() != 1) {
        throw UsageError("info expects exactly one input file");
    }
    if (verbose) {
        print_diagnostics();
    }

    const std::string& path = positional[0];
    const basalt::io::SegyData data = basalt::io::read_segy(path);
    const auto interval_us = static_cast<long long>(std::llround(data.dt * 1e6));

    std::cout << "path: " << path << "\n"
              << "ntraces: " << data.ntraces << "\n"
              << "nsamples: " << data.nsamples << "\n"
              << "dt: " << data.dt << " s (" << interval_us << " us)\n"
              << "format: " << format_name(data.format) << " (code "
              << static_cast<unsigned>(data.format) << ")\n"
              << "text header:\n";
    for (const std::string& line : leading_text_lines(data.text_header, 3)) {
        std::cout << "  " << line << "\n";
    }
    return kExitOk;
}

int run_filter(const std::vector<std::string>& args) {
    basalt::FKMuteParams params;
    std::optional<uint16_t> dt_us;                  // --dt, in microseconds
    size_t gather_size = 0;                         // --gather-size; 0 = whole file
    std::optional<basalt::io::SampleFormat> format; // --format
    bool verbose = false;
    std::vector<std::string> positional;

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg.empty() || arg[0] != '-') {
            positional.push_back(arg);
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            print_help(std::cout);
            return kExitOk;
        }
        if (arg == "--verbose") {
            verbose = true;
            continue;
        }
        const auto value = [&]() -> const std::string& {
            if (i + 1 >= args.size()) {
                throw UsageError("missing value for " + arg);
            }
            return args[++i];
        };

        if (arg == "--vmin") {
            params.v_min = parse_number(arg, value());
        } else if (arg == "--vmax") {
            params.v_max = parse_number(arg, value());
        } else if (arg == "--taper") {
            params.taper = parse_number(arg, value());
        } else if (arg == "--mode") {
            params.mode = parse_mode(value());
        } else if (arg == "--dx") {
            params.dx = parse_number(arg, value());
        } else if (arg == "--dt") {
            dt_us = dt_to_microseconds(parse_number(arg, value()));
        } else if (arg == "--pad") {
            params.pad_factor = parse_number(arg, value());
        } else if (arg == "--edge-taper") {
            params.edge_taper_traces = parse_count(arg, value());
        } else if (arg == "--threads") {
            params.threads = parse_count(arg, value());
        } else if (arg == "--gather-size") {
            gather_size = parse_positive_count(arg, value());
        } else if (arg == "--format") {
            format = parse_format(value());
        } else {
            throw UsageError("unknown option '" + arg + "'");
        }
    }

    if (positional.size() != 2) {
        throw UsageError("filter expects an input and an output file");
    }
    if (!(params.v_min < params.v_max)) {
        throw UsageError("--vmin must be less than --vmax");
    }
    if (!(params.taper >= 0.0 && params.taper < 1.0)) {
        throw UsageError("--taper must be in [0, 1)");
    }
    if (!(params.dx > 0.0)) {
        throw UsageError("--dx must be positive");
    }
    if (!(params.pad_factor >= 1.0)) {
        throw UsageError("--pad must be at least 1");
    }

    const std::string& in_path = positional[0];
    const std::string& out_path = positional[1];
    // Writing over the input would destroy it if the write failed part way, so refuse up front.
    std::error_code same_ec;
    if (in_path == out_path || std::filesystem::equivalent(in_path, out_path, same_ec)) {
        throw UsageError("input and output must be different files");
    }
    if (verbose) {
        print_diagnostics();
    }

    const auto read_started = std::chrono::steady_clock::now();
    basalt::io::SegyData data = basalt::io::read_segy(in_path, params.threads);
    const double read_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - read_started)
                               .count();
    if (data.ntraces == 0) {
        // The library rejects empty gathers; refuse before writing so no empty output is left.
        throw std::runtime_error("input has no traces to filter: " + in_path);
    }
    if (dt_us) {
        set_sample_interval(data, *dt_us);
        data.dt = static_cast<double>(*dt_us) * 1e-6;
    }
    params.dt = data.dt;
    const basalt::io::SampleFormat out_format =
        format ? *format : default_output_format(data.format);

    // Each block is filtered as its own gather, in place. The default block is the whole file.
    const size_t ntraces = data.ntraces;
    const size_t nsamples = data.nsamples;
    const size_t block = gather_size != 0 ? std::min(gather_size, ntraces) : ntraces;
    // The edge taper weights the first and last N traces of each block, and the requested N is
    // used as given for every block (FK_PIPELINE.md, section 4). For a block of `count` traces the
    // two tapers share a trace when N > count / 2. The contract does not define that case yet, so
    // the library multiplies the two weights there. A warning names each block size that overlaps.
    const size_t edge_taper = params.edge_taper_traces;
    const size_t tail = ntraces % block;
    if (edge_taper > block / 2) {
        std::cerr << "basalt: warning: --edge-taper " << edge_taper << " overlaps within " << block
                  << "-trace blocks (the start and end tapers multiply)\n";
    }
    if (tail != 0 && edge_taper > tail / 2) {
        std::cerr << "basalt: warning: --edge-taper " << edge_taper << " overlaps within the last "
                  << tail << "-trace block (the start and end tapers multiply)\n";
    }
    // With two or more gathers each gather is filtered on one thread, so the output does not depend
    // on --threads. A single gather is filtered with --threads threads, since there is nothing to
    // run it alongside.
    const size_t nfull = ntraces / block;
    const bool several_gathers = nfull + (tail != 0 ? 1 : 0) >= 2;
    basalt::FKMuteParams single = params;
    single.threads = several_gathers ? 1 : params.threads;
    const size_t workers = resolve_threads(params.threads);
    float* samples = data.samples.data();
    float* last = samples + nfull * block * nsamples;
    const auto started = std::chrono::steady_clock::now();
    if (!several_gathers) {
        basalt::fk_filter_gather(samples, samples, ntraces, nsamples, single);
    } else if (workers == 1) {
        // One thread: the gathers in turn. This gives the batch result (fk_filter_gathers with one
        // thread is fk_filter_gather per gather), without the batch's up-front non-finite scan.
        // That scan is not needed here, since a failed run writes no output file.
        for (size_t g = 0; g < nfull; ++g) {
            float* block_start = samples + g * block * nsamples;
            basalt::fk_filter_gather(block_start, block_start, block, nsamples, single);
        }
        if (tail != 0) {
            basalt::fk_filter_gather(last, last, tail, nsamples, single);
        }
    } else {
        // Several threads: the short last block runs on a helper thread while the full blocks run
        // here, each gather still on one thread. The two regions of `samples` do not overlap.
        std::thread helper;
        std::exception_ptr tail_error;
        bool overlapped = false;
        if (tail != 0) {
            try {
                helper = std::thread([last, tail, nsamples, &single, &tail_error] {
                    try {
                        basalt::fk_filter_gather(last, last, tail, nsamples, single);
                    } catch (...) {
                        tail_error = std::current_exception();
                    }
                });
                overlapped = true;
            } catch (const std::system_error&) {
                // No thread could be started: the tail is filtered below, after the full blocks.
            }
        }
        try {
            if (nfull == 1) {
                basalt::fk_filter_gather(samples, samples, block, nsamples, single);
            } else if (nfull >= 2) {
                // The helper, when running, takes one of the workers.
                basalt::FKMuteParams batch = params;
                batch.threads = overlapped ? workers - 1 : workers;
                basalt::fk_filter_gathers(samples, samples, nfull, block, nsamples, batch);
            }
            if (tail != 0 && !overlapped) {
                basalt::fk_filter_gather(last, last, tail, nsamples, single);
            }
        } catch (...) {
            if (helper.joinable()) {
                helper.join();
            }
            throw;
        }
        if (helper.joinable()) {
            helper.join();
        }
        if (tail_error) {
            std::rethrow_exception(tail_error);
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(finished - started).count();

    const auto write_started = std::chrono::steady_clock::now();
    basalt::io::write_segy(out_path, data, out_format, params.threads);
    const auto write_finished = std::chrono::steady_clock::now();
    if (verbose) {
        std::cerr << std::fixed << std::setprecision(3) << "basalt: read " << read_ms
                  << " ms, filter " << elapsed_ms << " ms, write "
                  << std::chrono::duration<double, std::milli>(write_finished - write_started)
                         .count()
                  << " ms\n";
    }

    std::cout << std::fixed << std::setprecision(3) << "filtered " << ntraces << " traces, "
              << nsamples << " samples in " << elapsed_ms << " ms -> " << out_path << " ("
              << format_name(out_format) << ")\n";
    return kExitOk;
}

} // namespace

int main(int argc, char** argv) {
    // Critical: set FTZ/DAZ before any floating-point computation.
    basalt::kernel::set_flush_to_zero();

    try {
        if (argc < 2) {
            throw UsageError("missing command");
        }
        const std::string command = argv[1];
        const std::vector<std::string> rest(argv + 2, argv + argc);

        if (command == "--help" || command == "-h") {
            print_help(std::cout);
            return kExitOk;
        }
        if (command == "--version") {
            std::cout << "basalt " << BASALT_VERSION << "\n";
            return kExitOk;
        }
        if (command == "info") {
            return run_info(rest);
        }
        if (command == "filter") {
            return run_filter(rest);
        }
        throw UsageError("unknown command '" + command + "'");
    } catch (const UsageError& e) {
        std::cerr << "basalt: " << e.what() << "\n";
        print_usage(std::cerr);
        return kExitUsage;
    } catch (const std::bad_alloc&) {
        std::cerr << "basalt: error: out of memory "
                     "(the padded FFT buffer is too large; try a smaller --pad)\n";
        return kExitRuntime;
    } catch (const std::exception& e) {
        std::cerr << "basalt: error: " << e.what() << "\n";
        return kExitRuntime;
    }
}
