// basalt: command line front end for the Basalt f-k library.
//
// Subcommands:
//   basalt info
//   basalt inspect <file.segy>
//   basalt filter <in.segy> <out.segy> --dx M [--dt S] [--vmin V] [--vmax V] [--taper W]
//
// Exit codes: 0 success, 1 runtime failure (bad file, I/O, out of memory),
// 2 usage error (unknown command, bad or missing arguments).

#include "basalt/arena.hpp"
#include "basalt/filter/fk_filter.hpp"
#include "basalt/io/file_mapping.hpp"
#include "basalt/io/segy_parser.hpp"
#include "basalt/io/segy_writer.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/cpu_features.hpp"
#include "basalt/kernel/cpu_state.hpp"
#include "basalt/kernel/fft2d.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#ifndef BASALT_VERSION
#define BASALT_VERSION "unknown"
#endif

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;

// Thrown for bad command lines; reported with exit code 2.
class UsageError : public std::runtime_error {
public:
    explicit UsageError(const std::string& what) : std::runtime_error(what) {}
};

void print_usage(std::ostream& os) {
    os << "Usage: basalt <command> [options]\n"
          "\n"
          "Commands:\n"
          "  info                         Show version, CPU features and FTZ/DAZ state\n"
          "  inspect <file.segy>          Print the layout of a SEG-Y file\n"
          "  filter <in.segy> <out.segy>  Apply an f-k velocity mute to a SEG-Y file\n"
          "\n"
          "filter options:\n"
          "  --dx M       Trace spacing in meters (required)\n"
          "  --dt S       Sample interval in seconds (default: from the binary header)\n"
          "  --vmin V     Lower edge of the muted apparent velocity band in m/s, v = f/k\n"
          "               (default -1500, or -vmax when only --vmax is given)\n"
          "  --vmax V     Upper edge of the muted band in m/s (default 1500)\n"
          "  --taper W    Width of the sigmoid band edges in m/s (default 100)\n"
          "  --max-memory MB  Refuse inputs needing more working memory than this\n"
          "                   (default: 75% of physical memory, 4096 if unknown)\n"
          "\n"
          "General options:\n"
          "  -h, --help   Show this help and exit\n"
          "  --version    Show the version and exit\n"
          "\n"
          "Exit status: 0 success, 1 runtime error, 2 usage error.\n";
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

void print_cpu_capabilities() {
    const basalt::kernel::CpuFeatures& cpu = basalt::kernel::cpu_features();
    const auto yn = [](bool supported) { return supported ? "yes" : "no"; };
    std::cout << "CPU features (runtime, after BASALT_SIMD_MAX):\n";
    std::cout << "  avx2:     " << yn(cpu.avx2) << "\n";
    std::cout << "  fma:      " << yn(cpu.fma) << "\n";
    std::cout << "  avx512f:  " << yn(cpu.avx512f) << "\n";
    std::cout << "  avx512dq: " << yn(cpu.avx512dq) << "\n";
    const char* path = cpu.can_use_avx512() ? "avx512" : cpu.can_use_avx2() ? "avx2" : "scalar";
    std::cout << "Kernel path: " << path << "\n";
}

int cmd_info(const std::vector<std::string>& args) {
    if (!args.empty()) {
        throw UsageError("'info' takes no arguments");
    }
    std::cout << "basalt " << BASALT_VERSION << "\n";
    print_cpu_capabilities();
    std::cout << "FTZ/DAZ: " << (basalt::kernel::check_ftz_daz() ? "enabled" : "disabled") << "\n";
    return kExitOk;
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

const char* format_name(uint16_t code) {
    switch (code) {
    case 1:
        return "IBM float, 32 bit";
    case 2:
        return "int32";
    case 3:
        return "int16";
    case 5:
        return "IEEE float, 32 bit";
    case 6:
        return "IEEE float, 64 bit";
    case 7:
        return "int24";
    case 8:
        return "int8";
    case 9:
        return "int64";
    case 10:
        return "uint32";
    case 11:
        return "uint16";
    case 12:
        return "uint64";
    case 15:
        return "uint24";
    case 16:
        return "uint8";
    default:
        return "unknown";
    }
}

// Binary header interval, falling back to the first trace header when unset.
uint32_t sample_interval_us(const basalt::io::SegyLayout& lay, const uint8_t* data, size_t size) {
    if (lay.binary.sample_interval != 0) {
        return lay.binary.sample_interval;
    }
    if (lay.n_traces > 0) {
        return basalt::io::read_trace_header(data, size, lay, 0).sample_interval_us;
    }
    return 0;
}

void print_trace_keys(const char* label, const basalt::SegyTraceHeader& h) {
    std::cout << label;
    if (h.inline_number == 0 && h.crossline_number == 0 && h.cdp_ensemble_number == 0) {
        std::cout << "(inline, crossline and cdp not set)\n";
        return;
    }
    std::cout << "inline=" << h.inline_number << " crossline=" << h.crossline_number
              << " cdp=" << h.cdp_ensemble_number << "\n";
}

int cmd_inspect(const std::vector<std::string>& args) {
    if (args.size() != 1) {
        throw UsageError("'inspect' needs exactly one argument: <file.segy>");
    }
    const std::string& path = args[0];
    basalt::io::FileMapping map(path);
    const auto* data = static_cast<const uint8_t*>(map.data());
    const basalt::io::SegyLayout lay = basalt::io::describe_segy(data, map.size());

    const uint32_t dt_us = sample_interval_us(lay, data, map.size());

    std::cout << "File:               " << path << "\n";
    std::cout << "Size:               " << map.size() << " bytes\n";
    std::cout << "SEG-Y revision:     " << static_cast<unsigned>(lay.revision_major) << "."
              << static_cast<unsigned>(lay.revision_minor) << "\n";
    std::cout << "Data format:        " << lay.data_sample_format << " ("
              << format_name(lay.data_sample_format) << ")\n";
    std::cout << "Byte order:         " << (lay.little_endian ? "little-endian" : "big-endian")
              << "\n";
    if (lay.variable_length) {
        std::cout << "Variable-length traces: yes\n";
        std::cout << "Min samples/trace:  " << lay.min_samples_per_trace << "\n";
        std::cout << "Max samples/trace:  " << lay.samples_per_trace << "\n";
    } else {
        std::cout << "Samples per trace:  " << lay.samples_per_trace << "\n";
        std::cout << "Variable-length traces: no\n";
    }
    std::cout << "Sample interval:    " << dt_us << " us";
    if (dt_us != 0) {
        std::cout << " (" << static_cast<double>(dt_us) * 1e-6 << " s)";
    }
    std::cout << "\n";
    std::cout << "Traces:             " << lay.n_traces << "\n";
    std::cout << "Extended headers:   " << lay.n_ext_text_headers << "\n";
    if (lay.trailer_bytes != 0) {
        std::cout << "Trailer bytes:      " << lay.trailer_bytes << "\n";
    }
    if (lay.n_traces > 0) {
        print_trace_keys("First trace:        ",
                         basalt::io::read_trace_header(data, map.size(), lay, 0));
        print_trace_keys("Last trace:         ",
                         basalt::io::read_trace_header(data, map.size(), lay, lay.n_traces - 1));
    }
    return kExitOk;
}

// ---------------------------------------------------------------------------
// filter
// ---------------------------------------------------------------------------

double parse_number(const std::string& opt, const std::string& text) {
    if (text.empty()) {
        throw UsageError("option " + opt + " needs a numeric value");
    }
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
        throw UsageError("option " + opt + ": '" + text + "' is not a valid finite number");
    }
    return v;
}

struct FilterArgs {
    std::string in_path;
    std::string out_path;
    std::optional<double> dx;
    std::optional<double> dt;
    std::optional<double> vmin;
    std::optional<double> vmax;
    std::optional<double> taper;
    std::optional<double> max_memory_mb;
};

FilterArgs parse_filter_args(const std::vector<std::string>& args) {
    FilterArgs fa;
    std::vector<std::string> positional;
    for (size_t i = 0; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg.size() < 2 || arg[0] != '-' || arg[1] != '-') {
            if (arg.size() >= 2 && arg[0] == '-') {
                throw UsageError("unknown option '" + arg + "'");
            }
            positional.push_back(arg);
            continue;
        }
        std::string name = arg;
        std::optional<std::string> value;
        const size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            name = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }
        std::optional<double>* target = nullptr;
        if (name == "--dx")
            target = &fa.dx;
        else if (name == "--dt")
            target = &fa.dt;
        else if (name == "--vmin")
            target = &fa.vmin;
        else if (name == "--vmax")
            target = &fa.vmax;
        else if (name == "--taper")
            target = &fa.taper;
        else if (name == "--max-memory")
            target = &fa.max_memory_mb;
        else
            throw UsageError("unknown option '" + name + "'");

        if (!value) {
            if (i + 1 >= args.size()) {
                throw UsageError("option " + name + " needs a value");
            }
            value = args[++i];
        }
        *target = parse_number(name, *value);
    }
    if (positional.size() != 2) {
        throw UsageError("'filter' needs exactly two files: <in.segy> <out.segy>");
    }
    fa.in_path = positional[0];
    fa.out_path = positional[1];
    return fa;
}

// Smallest power of two >= n (n >= 1); throws if it would exceed 2^30.
size_t next_pow2(size_t n, const char* what) {
    constexpr size_t kLimit = size_t{1} << 30;
    if (n > kLimit) {
        throw std::runtime_error(std::string(what) + " count " + std::to_string(n) +
                                 " is too large to transform");
    }
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

// Default working-memory budget in bytes.
uint64_t default_memory_budget() {
    uint64_t physical = 0;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        physical = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    }
#endif
    if (physical == 0) {
        return uint64_t{4096} << 20;
    }
    return physical / 4 * 3;
}

std::string megabytes(uint64_t bytes) {
    return std::to_string((bytes + (uint64_t{1} << 20) - 1) >> 20) + " MiB";
}

int cmd_filter(const std::vector<std::string>& args) {
    const FilterArgs fa = parse_filter_args(args);

    if (!fa.dx) {
        throw UsageError("--dx is required: the trace spacing in meters is not stored in the file");
    }
    basalt::filter::FKParams params;
    params.dx = *fa.dx;
    params.mute_vel_max = fa.vmax.value_or(1500.0);
    params.mute_vel_min = fa.vmin.value_or(fa.vmax ? -params.mute_vel_max : -1500.0);
    params.taper_width = fa.taper.value_or(100.0);
    if (!(params.dx > 0.0)) {
        throw UsageError("--dx must be > 0");
    }
    if (!(params.taper_width > 0.0)) {
        throw UsageError("--taper must be > 0");
    }
    if (!(params.mute_vel_min < params.mute_vel_max)) {
        throw UsageError("--vmin must be smaller than --vmax");
    }
    if (fa.dt && !(*fa.dt > 0.0)) {
        throw UsageError("--dt must be > 0");
    }
    if (fa.max_memory_mb && !(*fa.max_memory_mb > 0.0)) {
        throw UsageError("--max-memory must be > 0");
    }

    std::error_code ec;
    if (std::filesystem::exists(fa.out_path, ec) &&
        std::filesystem::equivalent(fa.in_path, fa.out_path, ec)) {
        throw UsageError("input and output must be different files");
    }

    basalt::io::FileMapping map(fa.in_path);
    const auto* data = static_cast<const uint8_t*>(map.data());
    const size_t size = map.size();
    const basalt::io::SegyLayout lay = basalt::io::describe_segy(data, size);
    if (lay.n_traces == 0) {
        throw std::runtime_error("input has no traces");
    }

    if (fa.dt) {
        params.dt = *fa.dt;
    } else {
        const uint32_t us = sample_interval_us(lay, data, size);
        if (us == 0) {
            throw UsageError("the file has no sample interval; pass --dt in seconds");
        }
        params.dt = static_cast<double>(us) * 1e-6;
    }

    // Gather geometry: rows = time samples (the f axis after the FFT), cols =
    // traces (the k axis). Element (sample i, trace j) lives at i * cols + j,
    // which is the orientation apply_fk_filter expects.
    // For variable-length files ns is the longest trace: shorter traces are
    // zero-padded up to it for the transform and cropped back on output.
    const size_t ns = lay.samples_per_trace;
    const size_t nt = lay.n_traces;
    if (ns == 0) {
        throw std::runtime_error("input traces contain no samples");
    }
    const size_t rows = next_pow2(ns, "sample");
    const size_t cols = next_pow2(nt, "trace");
    if (rows > std::numeric_limits<size_t>::max() / cols) {
        throw std::runtime_error("padded gather is too large to transform");
    }
    const size_t total = rows * cols;

    const size_t soa_bytes = basalt::kernel::ComplexSoA::required_arena_bytes(total);
    const size_t scratch_bytes = basalt::kernel::fft2d_scratch_bytes(rows, cols);
    const uint64_t budget =
        fa.max_memory_mb ? static_cast<uint64_t>(std::min(*fa.max_memory_mb, 1e9) * 1048576.0)
                         : default_memory_budget();
    const uint64_t kMax = std::numeric_limits<size_t>::max();
    const bool overflow =
        soa_bytes == kMax || scratch_bytes == kMax || soa_bytes > kMax - scratch_bytes;
    if (overflow || static_cast<uint64_t>(soa_bytes) + scratch_bytes > budget) {
        const std::string need = overflow
                                     ? std::string("more than the address space")
                                     : megabytes(static_cast<uint64_t>(soa_bytes) + scratch_bytes);
        throw std::runtime_error("gather of " + std::to_string(ns) + " samples x " +
                                 std::to_string(nt) + " traces (padded to " + std::to_string(rows) +
                                 " x " + std::to_string(cols) + ") needs " + need +
                                 " of working memory, above the limit of " + megabytes(budget) +
                                 "; use --max-memory to raise the limit");
    }

    basalt::MemoryArena data_arena(soa_bytes);
    basalt::MemoryArena scratch(scratch_bytes);
    basalt::kernel::ComplexSoA gather(data_arena, total);
    gather.clear();

    // Load and de-interleave: trace j fills column j.
    std::vector<float> trace(ns);
    for (size_t j = 0; j < nt; ++j) {
        const size_t nj = lay.trace_samples(j);
        basalt::io::read_trace_samples(data, size, lay, j, trace.data(), trace.size());
        for (size_t i = 0; i < nj; ++i) {
            const float v = trace[i];
            if (!std::isfinite(v)) {
                throw std::runtime_error("trace " + std::to_string(j) + " sample " +
                                         std::to_string(i) + " is not finite (NaN or Inf)");
            }
            gather.real[i * cols + j] = v;
        }
    }

    basalt::kernel::fft2d_forward(gather.real, gather.imag, rows, cols, scratch);
    basalt::filter::apply_fk_filter(gather, rows, cols, params);
    basalt::kernel::fft2d_inverse(gather.real, gather.imag, rows, cols, scratch);

    // Crop and write with the original headers.
    basalt::io::SegyWriter writer(fa.out_path, data, lay.data_offset, lay);
    for (size_t j = 0; j < nt; ++j) {
        const size_t nj = lay.trace_samples(j);
        for (size_t i = 0; i < nj; ++i) {
            trace[i] = gather.real[i * cols + j];
        }
        writer.write_trace(data + lay.trace_offset(j), trace.data());
    }
    // The traces tile the file up to the trailer stanzas (rev 2), if any.
    writer.commit(data + size - lay.trailer_bytes, lay.trailer_bytes);

    std::cout << "Filtered " << nt << " traces x " << ns << " samples (dt=" << params.dt
              << " s, dx=" << params.dx << " m), padded to " << rows << " x " << cols << "\n";
    if (lay.variable_length) {
        std::cout << "Variable-length traces (" << lay.min_samples_per_trace << " to " << ns
                  << " samples): zero-padded to " << ns
                  << " samples for the transform, each trace cropped back to its own length\n";
    }
    std::cout << "Muted apparent velocities " << params.mute_vel_min << " to "
              << params.mute_vel_max << " m/s, taper " << params.taper_width << " m/s\n";
    if (writer.output_format() != lay.data_sample_format) {
        std::cout << "Note: input sample format " << lay.data_sample_format
                  << " was written as IEEE float (format 5)\n";
    }
    std::cout << "Wrote " << fa.out_path << "\n";
    return kExitOk;
}

} // namespace

int main(int argc, char* argv[]) {
    // Critical: set FTZ/DAZ before any floating-point computation.
    basalt::kernel::set_flush_to_zero();

    const std::vector<std::string> argv_all(argv + 1, argv + argc);
    try {
        if (argv_all.empty()) {
            print_usage(std::cerr);
            return kExitUsage;
        }
        const std::string& cmd = argv_all[0];
        const std::vector<std::string> rest(argv_all.begin() + 1, argv_all.end());
        if (std::find_if(rest.begin(), rest.end(),
                         [](const std::string& a) { return a == "-h" || a == "--help"; }) !=
                rest.end() &&
            (cmd == "info" || cmd == "inspect" || cmd == "filter")) {
            print_usage(std::cout);
            return kExitOk;
        }

        if (cmd == "-h" || cmd == "--help" || cmd == "help") {
            print_usage(std::cout);
            return kExitOk;
        }
        if (cmd == "--version") {
            std::cout << "basalt " << BASALT_VERSION << "\n";
            return kExitOk;
        }
        if (cmd == "info")
            return cmd_info(rest);
        if (cmd == "inspect")
            return cmd_inspect(rest);
        if (cmd == "filter")
            return cmd_filter(rest);
        throw UsageError("unknown command '" + cmd + "'");
    } catch (const UsageError& e) {
        std::cerr << "basalt: error: " << e.what() << "\n\n";
        print_usage(std::cerr);
        return kExitUsage;
    } catch (const basalt::io::SegyFormatError& e) {
        std::cerr << "basalt: error: invalid SEG-Y input: " << e.what() << "\n";
        return kExitFailure;
    } catch (const std::bad_alloc&) {
        std::cerr << "basalt: error: out of memory\n";
        return kExitFailure;
    } catch (const std::exception& e) {
        std::cerr << "basalt: error: " << e.what() << "\n";
        return kExitFailure;
    }
}
