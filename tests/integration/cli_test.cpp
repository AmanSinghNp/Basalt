#include <gtest/gtest.h>

#ifdef BASALT_CLI_PATH

#include "basalt/io/segy_parser.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define BASALT_TEST_GETPID _getpid
#else
#include <sys/wait.h>
#include <unistd.h>
#define BASALT_TEST_GETPID getpid
#endif

using namespace basalt::io;
namespace fs = std::filesystem;

namespace {

using Bytes = std::vector<uint8_t>;

struct RunResult {
    int exit_code = -1;
    std::string output; // stdout and stderr combined
};

std::string quote(const std::string& s) {
    return "\"" + s + "\"";
}

RunResult run_cli(const std::string& args) {
    RunResult r;
    const std::string cmd = quote(BASALT_CLI_PATH) + " " + args + " 2>&1";
#if defined(_WIN32)
    FILE* pipe = _popen(("\"" + cmd + "\"").c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe)
        return r;
    char buf[512];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, pipe)) > 0) {
        r.output.append(buf, n);
    }
#if defined(_WIN32)
    r.exit_code = _pclose(pipe);
#else
    const int status = pclose(pipe);
    r.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    return r;
}

class TempDir {
public:
    TempDir() {
        static int counter = 0;
        m_path = fs::temp_directory_path() /
                 ("basalt_cli_test_" + std::to_string(static_cast<long>(BASALT_TEST_GETPID())) +
                  "_" + std::to_string(counter++));
        fs::remove_all(m_path);
        fs::create_directories(m_path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(m_path, ec);
    }
    [[nodiscard]] std::string file(const std::string& name) const {
        return (m_path / name).string();
    }

private:
    fs::path m_path;
};

void put(Bytes& b, size_t off, uint64_t value, size_t width) {
    if (off + width > b.size()) {
        ADD_FAILURE() << "put out of range";
        return;
    }
    for (size_t i = 0; i < width; ++i) {
        b[off + i] = static_cast<uint8_t>((value >> (8 * (width - 1 - i))) & 0xFF);
    }
}

constexpr double kPi = 3.14159265358979323846;

// 10 Hz Ricker wavelet value at time offset t from its peak.
double ricker(double t) {
    const double a = kPi * 10.0 * t;
    return (1.0 - 2.0 * a * a) * std::exp(-a * a);
}

struct Gather {
    size_t ns = 512;
    size_t nt = 64;
    double dt = 0.002;
    double dx = 5.0;
    std::vector<float> samples; // trace-major: samples[trace * ns + sample]
};

// Slow dipping event (500 m/s from t0 = 0.05 s) plus a flat event at 0.9 s.
Gather make_gather() {
    Gather g;
    g.samples.assign(g.ns * g.nt, 0.0f);
    for (size_t j = 0; j < g.nt; ++j) {
        const double arrival = 0.05 + static_cast<double>(j) * g.dx / 500.0;
        for (size_t i = 0; i < g.ns; ++i) {
            const double t = static_cast<double>(i) * g.dt;
            g.samples[j * g.ns + i] = static_cast<float>(ricker(t - arrival) + ricker(t - 0.9));
        }
    }
    return g;
}

Bytes build_segy(const Gather& g) {
    const size_t stride = 240 + g.ns * 4;
    Bytes b(3600 + g.nt * stride, 0);
    std::fill(b.begin(), b.begin() + 3200, uint8_t{0x40}); // EBCDIC blanks
    put(b, 3200 + 16, static_cast<uint64_t>(std::lround(g.dt * 1e6)), 2);
    put(b, 3200 + 20, g.ns, 2);
    put(b, 3200 + 24, 5, 2); // IEEE float
    b[3200 + 300] = 1;       // revision 1.0
    put(b, 3200 + 302, 1, 2);
    for (size_t j = 0; j < g.nt; ++j) {
        const size_t off = 3600 + j * stride;
        put(b, off + 0, j + 1, 4);     // trace_seq_line
        put(b, off + 20, 1000 + j, 4); // cdp_ensemble_number
        put(b, off + 114, g.ns, 2);
        put(b, off + 116, static_cast<uint64_t>(std::lround(g.dt * 1e6)), 2);
        put(b, off + 188, 100, 4);     // inline
        put(b, off + 192, 200 + j, 4); // crossline
        for (size_t i = 0; i < g.ns; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &g.samples[j * g.ns + i], 4);
            put(b, off + 240 + 4 * i, bits, 4);
        }
    }
    return b;
}

// Variable-length rev 1 file (fixed-length flag 0): trace j keeps only its first
// counts[j] samples of 'g' and says so in its header (bytes 115-116). The
// binary header ns is the maximum, as many writers do.
Bytes build_variable_segy(const Gather& g, const std::vector<size_t>& counts) {
    size_t body = 0;
    for (size_t c : counts) {
        body += 240 + c * 4;
    }
    Bytes b(3600 + body, 0);
    std::fill(b.begin(), b.begin() + 3200, uint8_t{0x40});
    put(b, 3200 + 16, static_cast<uint64_t>(std::lround(g.dt * 1e6)), 2);
    put(b, 3200 + 20, *std::max_element(counts.begin(), counts.end()), 2);
    put(b, 3200 + 24, 5, 2);
    b[3200 + 300] = 1;
    put(b, 3200 + 302, 0, 2);
    size_t off = 3600;
    for (size_t j = 0; j < counts.size(); ++j) {
        put(b, off + 0, j + 1, 4);
        put(b, off + 20, 1000 + j, 4);
        put(b, off + 114, counts[j], 2);
        put(b, off + 116, static_cast<uint64_t>(std::lround(g.dt * 1e6)), 2);
        put(b, off + 188, 100, 4);
        put(b, off + 192, 200 + j, 4);
        for (size_t i = 0; i < counts[j]; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &g.samples[j * g.ns + i], 4);
            put(b, off + 240 + 4 * i, bits, 4);
        }
        off += 240 + counts[j] * 4;
    }
    return b;
}

// Per-trace lengths between 300 and 512 (maximum reached), never all equal.
std::vector<size_t> variable_counts(size_t nt) {
    std::vector<size_t> c(nt);
    for (size_t j = 0; j < nt; ++j) {
        c[j] = 300 + (j * 37) % 213; // 300..512
    }
    c[0] = 512;
    c[1] = 300;
    return c;
}

void write_file(const std::string& path, const Bytes& b) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

Bytes read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// RMS over trace-major samples with time index in [i0, i1) across all traces.
double rms(const std::vector<float>& v, size_t ns, size_t nt, size_t i0, size_t i1) {
    double sum = 0.0;
    size_t n = 0;
    for (size_t j = 0; j < nt; ++j) {
        for (size_t i = i0; i < i1; ++i) {
            sum += static_cast<double>(v[j * ns + i]) * v[j * ns + i];
            ++n;
        }
    }
    return std::sqrt(sum / static_cast<double>(n));
}

} // namespace

TEST(CliTest, HelpAndVersionExitZero) {
    for (const char* a : {"--help", "-h", "filter --help"}) {
        const RunResult r = run_cli(a);
        EXPECT_EQ(r.exit_code, 0) << a;
        EXPECT_NE(r.output.find("Usage:"), std::string::npos) << a;
    }
    const RunResult v = run_cli("--version");
    EXPECT_EQ(v.exit_code, 0);
    EXPECT_NE(v.output.find("basalt"), std::string::npos);
}

TEST(CliTest, BadArgumentsExitTwo) {
    TempDir dir;
    EXPECT_EQ(run_cli("").exit_code, 2);
    EXPECT_EQ(run_cli("frobnicate").exit_code, 2);
    EXPECT_EQ(run_cli("info extra").exit_code, 2);
    EXPECT_EQ(run_cli("inspect").exit_code, 2);
    EXPECT_EQ(run_cli("inspect a b").exit_code, 2);
    EXPECT_EQ(run_cli("filter in.segy").exit_code, 2);
    const RunResult missing_dx =
        run_cli("filter " + quote(dir.file("a")) + " " + quote(dir.file("b")));
    EXPECT_EQ(missing_dx.exit_code, 2);
    EXPECT_NE(missing_dx.output.find("--dx"), std::string::npos);
    EXPECT_EQ(run_cli("filter a b --dx").exit_code, 2);
    EXPECT_EQ(run_cli("filter a b --dx abc").exit_code, 2);
    EXPECT_EQ(run_cli("filter a b --dx 5 --bogus 1").exit_code, 2);
    EXPECT_EQ(run_cli("filter a b --dx 5 --vmin 100 --vmax 50").exit_code, 2);
    EXPECT_EQ(run_cli("filter a b --dx 5 --taper 0").exit_code, 2);
    // Usage goes to stderr, which run_cli merges; make sure the message is present.
    EXPECT_NE(run_cli("frobnicate").output.find("unknown command"), std::string::npos);
}

TEST(CliTest, InfoReportsFtzAndFeatures) {
    const RunResult r = run_cli("info");
    EXPECT_EQ(r.exit_code, 0);
    EXPECT_NE(r.output.find("FTZ/DAZ"), std::string::npos);
    EXPECT_NE(r.output.find("avx2"), std::string::npos);
}

TEST(CliTest, InspectPrintsLayout) {
    TempDir dir;
    const Gather g = make_gather();
    const std::string path = dir.file("in.segy");
    write_file(path, build_segy(g));

    const RunResult r = run_cli("inspect " + quote(path));
    ASSERT_EQ(r.exit_code, 0) << r.output;
    EXPECT_NE(r.output.find("SEG-Y revision:     1.0"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("Data format:        5"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("big-endian"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("Samples per trace:  512"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("2000 us"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("Traces:             64"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("Extended headers:   0"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("inline=100 crossline=200 cdp=1000"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("inline=100 crossline=263 cdp=1063"), std::string::npos) << r.output;
}

TEST(CliTest, InspectReportsBadFilesWithExitOne) {
    TempDir dir;
    EXPECT_EQ(run_cli("inspect " + quote(dir.file("missing.segy"))).exit_code, 1);

    const std::string junk = dir.file("junk.segy");
    write_file(junk, Bytes(100, 0x55));
    const RunResult r = run_cli("inspect " + quote(junk));
    EXPECT_EQ(r.exit_code, 1);
    EXPECT_NE(r.output.find("error"), std::string::npos);

    // Truncated by a partial trace.
    Bytes b = build_segy(make_gather());
    b.resize(b.size() - 10);
    const std::string cut = dir.file("cut.segy");
    write_file(cut, b);
    EXPECT_EQ(run_cli("inspect " + quote(cut)).exit_code, 1);
    EXPECT_EQ(
        run_cli("filter " + quote(cut) + " " + quote(dir.file("o.segy")) + " --dx 5").exit_code, 1);
    EXPECT_FALSE(fs::exists(dir.file("o.segy")));
}

TEST(CliTest, FilterAttenuatesSlowEventAndKeepsFlatEvent) {
    TempDir dir;
    const Gather g = make_gather();
    const std::string in_path = dir.file("in.segy");
    const std::string out_path = dir.file("out.segy");
    const Bytes src = build_segy(g);
    write_file(in_path, src);

    const RunResult r = run_cli("filter " + quote(in_path) + " " + quote(out_path) +
                                " --dx 5 --vmax 1500 --taper 200");
    ASSERT_EQ(r.exit_code, 0) << r.output;
    EXPECT_NE(r.output.find("Filtered 64 traces x 512 samples"), std::string::npos) << r.output;
    ASSERT_TRUE(fs::exists(out_path));
    // No temporary files left next to the output.
    size_t entries = 0;
    for (const auto& e : fs::directory_iterator(fs::path(out_path).parent_path())) {
        (void)e;
        ++entries;
    }
    EXPECT_EQ(entries, 2u);

    const Bytes out = read_file(out_path);
    const SegyLayout in_lay = describe_segy(src.data(), src.size());
    const SegyLayout out_lay = describe_segy(out.data(), out.size());
    ASSERT_EQ(out_lay.n_traces, g.nt);
    ASSERT_EQ(out_lay.samples_per_trace, g.ns);
    ASSERT_EQ(out.size(), src.size());

    // Headers (file header and every trace header) are identical.
    ASSERT_EQ(0, std::memcmp(src.data(), out.data(), in_lay.data_offset));
    std::vector<float> filtered(g.ns * g.nt);
    for (size_t j = 0; j < g.nt; ++j) {
        ASSERT_EQ(0, std::memcmp(src.data() + in_lay.data_offset + j * in_lay.trace_stride,
                                 out.data() + out_lay.data_offset + j * out_lay.trace_stride, 240))
            << "trace header " << j;
        const auto s = read_trace_samples(out.data(), out.size(), out_lay, j);
        std::copy(s.begin(), s.end(), filtered.begin() + static_cast<std::ptrdiff_t>(j * g.ns));
    }

    // Time windows (sample index): the slow event occupies 0.05 s .. ~0.78 s
    // (samples 25..390); the flat event is at 0.9 s (sample 450).
    const double slow_in = rms(g.samples, g.ns, g.nt, 0, 400);
    const double slow_out = rms(filtered, g.ns, g.nt, 0, 400);
    const double flat_in = rms(g.samples, g.ns, g.nt, 400, 512);
    const double flat_out = rms(filtered, g.ns, g.nt, 400, 512);
    ASSERT_GT(slow_in, 1e-3);
    ASSERT_GT(flat_in, 1e-3);
    EXPECT_LT(slow_out / slow_in, 0.1) << "slow event not attenuated (>20 dB expected)";
    EXPECT_GT(flat_out / flat_in, 0.9) << "flat event was not preserved";
    EXPECT_LT(flat_out / flat_in, 1.1);
}

TEST(CliTest, FilterPadsNonPowerOfTwoGathersAndTakesDtOverride) {
    TempDir dir;
    Gather g;
    g.ns = 300;
    g.nt = 37;
    g.samples.assign(g.ns * g.nt, 0.0f);
    for (size_t j = 0; j < g.nt; ++j) {
        for (size_t i = 0; i < g.ns; ++i) {
            g.samples[j * g.ns + i] =
                static_cast<float>(ricker((static_cast<double>(i) - 150.0) * g.dt));
        }
    }
    const std::string in_path = dir.file("in.segy");
    const std::string out_path = dir.file("out.segy");
    write_file(in_path, build_segy(g));

    const RunResult r = run_cli("filter " + quote(in_path) + " " + quote(out_path) +
                                " --dx 10 --dt 0.004 --vmin=-800 --vmax=800");
    ASSERT_EQ(r.exit_code, 0) << r.output;
    EXPECT_NE(r.output.find("padded to 512 x 64"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("dt=0.004"), std::string::npos) << r.output;
    const Bytes out = read_file(out_path);
    const SegyLayout lay = describe_segy(out.data(), out.size());
    EXPECT_EQ(lay.n_traces, 37u);
    EXPECT_EQ(lay.samples_per_trace, 300u);
}

TEST(CliTest, InspectReportsVariableLengthTraces) {
    TempDir dir;
    const Gather g = make_gather();
    const std::vector<size_t> counts = variable_counts(g.nt);
    const std::string vpath = dir.file("var.segy");
    write_file(vpath, build_variable_segy(g, counts));

    const RunResult v = run_cli("inspect " + quote(vpath));
    ASSERT_EQ(v.exit_code, 0) << v.output;
    EXPECT_NE(v.output.find("Variable-length traces: yes"), std::string::npos) << v.output;
    EXPECT_NE(v.output.find("Min samples/trace:  300"), std::string::npos) << v.output;
    EXPECT_NE(v.output.find("Max samples/trace:  512"), std::string::npos) << v.output;
    EXPECT_NE(v.output.find("Traces:             64"), std::string::npos) << v.output;
    EXPECT_NE(v.output.find("inline=100 crossline=263 cdp=1063"), std::string::npos) << v.output;

    // A fixed-length file with the flag left at 0 is not variable.
    Bytes fixed = build_segy(g);
    put(fixed, 3200 + 302, 0, 2);
    const std::string fpath = dir.file("fixed.segy");
    write_file(fpath, fixed);
    const RunResult f = run_cli("inspect " + quote(fpath));
    ASSERT_EQ(f.exit_code, 0) << f.output;
    EXPECT_NE(f.output.find("Variable-length traces: no"), std::string::npos) << f.output;
    EXPECT_NE(f.output.find("Samples per trace:  512"), std::string::npos) << f.output;
    EXPECT_EQ(f.output.find("Min samples/trace"), std::string::npos) << f.output;
}

TEST(CliTest, FilterVariableLengthPadsThenCropsEachTrace) {
    TempDir dir;
    const Gather g = make_gather();
    const std::vector<size_t> counts = variable_counts(g.nt);
    const Bytes src = build_variable_segy(g, counts);
    const std::string in_path = dir.file("var.segy");
    const std::string out_path = dir.file("out.segy");
    write_file(in_path, src);

    const RunResult r = run_cli("filter " + quote(in_path) + " " + quote(out_path) +
                                " --dx 5 --vmax 1500 --taper 200");
    ASSERT_EQ(r.exit_code, 0) << r.output;
    EXPECT_NE(r.output.find("Filtered 64 traces x 512 samples"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("Variable-length traces (300 to 512 samples)"), std::string::npos)
        << r.output;
    EXPECT_NE(r.output.find("cropped back to its own length"), std::string::npos) << r.output;

    const Bytes out = read_file(out_path);
    const SegyLayout in_lay = describe_segy(src.data(), src.size());
    const SegyLayout out_lay = describe_segy(out.data(), out.size());
    ASSERT_TRUE(out_lay.variable_length);
    ASSERT_EQ(out_lay.n_traces, g.nt);
    ASSERT_EQ(out.size(), src.size()); // same per-trace lengths, same sample format
    ASSERT_EQ(0, std::memcmp(src.data(), out.data(), in_lay.data_offset));

    // Reference: the same data zero-padded to a fixed 512-sample file, filtered
    // with the same parameters. The variable result must equal it trace by
    // trace over each trace's own length.
    Gather padded = g;
    for (size_t j = 0; j < g.nt; ++j) {
        for (size_t i = counts[j]; i < g.ns; ++i) {
            padded.samples[j * g.ns + i] = 0.0f;
        }
    }
    const std::string ref_in = dir.file("padded.segy");
    const std::string ref_out = dir.file("padded_out.segy");
    write_file(ref_in, build_segy(padded));
    ASSERT_EQ(run_cli("filter " + quote(ref_in) + " " + quote(ref_out) +
                      " --dx 5 --vmax 1500 --taper 200")
                  .exit_code,
              0);
    const Bytes ref = read_file(ref_out);
    const SegyLayout ref_lay = describe_segy(ref.data(), ref.size());

    for (size_t j = 0; j < g.nt; ++j) {
        ASSERT_EQ(out_lay.trace_samples(j), counts[j]);
        ASSERT_EQ(0, std::memcmp(src.data() + in_lay.trace_offset(j),
                                 out.data() + out_lay.trace_offset(j), 240))
            << "trace header " << j;
        const auto got = read_trace_samples(out.data(), out.size(), out_lay, j);
        const auto want = read_trace_samples(ref.data(), ref.size(), ref_lay, j);
        ASSERT_EQ(got.size(), counts[j]);
        for (size_t i = 0; i < got.size(); ++i) {
            ASSERT_EQ(got[i], want[i]) << "trace " << j << " sample " << i;
        }
    }

    // The slow event is still attenuated (trace-major rows of 512; the cropped
    // tail of short traces is zero in the input and left out of the comparison).
    std::vector<float> in_rows(g.ns * g.nt, 0.0f);
    std::vector<float> out_rows(g.ns * g.nt, 0.0f);
    for (size_t j = 0; j < g.nt; ++j) {
        const auto a = read_trace_samples(src.data(), src.size(), in_lay, j);
        const auto b = read_trace_samples(out.data(), out.size(), out_lay, j);
        std::copy(a.begin(), a.end(), in_rows.begin() + static_cast<std::ptrdiff_t>(j * g.ns));
        std::copy(b.begin(), b.end(), out_rows.begin() + static_cast<std::ptrdiff_t>(j * g.ns));
    }
    const double slow_in = rms(in_rows, g.ns, g.nt, 0, 300);
    const double slow_out = rms(out_rows, g.ns, g.nt, 0, 300);
    ASSERT_GT(slow_in, 1e-3);
    EXPECT_LT(slow_out / slow_in, 0.3);
}

TEST(CliTest, VariableLengthFilesWithDamageFailCleanly) {
    TempDir dir;
    const Gather g = make_gather();
    const std::vector<size_t> counts = variable_counts(g.nt);
    Bytes b = build_variable_segy(g, counts);
    b.resize(b.size() - 10); // last trace short
    const std::string cut = dir.file("cut.segy");
    write_file(cut, b);
    EXPECT_EQ(run_cli("inspect " + quote(cut)).exit_code, 1);
    EXPECT_EQ(
        run_cli("filter " + quote(cut) + " " + quote(dir.file("o.segy")) + " --dx 5").exit_code, 1);
    EXPECT_FALSE(fs::exists(dir.file("o.segy")));

    // A per-trace count pointing far past the end of the file.
    Bytes evil = build_variable_segy(g, counts);
    put(evil, 3600 + 114, 65535, 2);
    const std::string evil_path = dir.file("evil.segy");
    write_file(evil_path, evil);
    const RunResult r = run_cli("inspect " + quote(evil_path));
    EXPECT_EQ(r.exit_code, 1);
    EXPECT_NE(r.output.find("invalid SEG-Y"), std::string::npos) << r.output;
}

TEST(CliTest, FilterRefusesToExceedMemoryLimitAndSameFile) {
    TempDir dir;
    const Gather g = make_gather();
    const std::string in_path = dir.file("in.segy");
    write_file(in_path, build_segy(g));

    const RunResult r = run_cli("filter " + quote(in_path) + " " + quote(dir.file("o.segy")) +
                                " --dx 5 --max-memory 0.25");
    EXPECT_EQ(r.exit_code, 1);
    EXPECT_NE(r.output.find("working memory"), std::string::npos) << r.output;
    EXPECT_FALSE(fs::exists(dir.file("o.segy")));

    const RunResult same = run_cli("filter " + quote(in_path) + " " + quote(in_path) + " --dx 5");
    EXPECT_EQ(same.exit_code, 2);
}

#else

TEST(CliTest, SkippedWithoutCliTarget) {
    GTEST_SKIP() << "BASALT_CLI_PATH not defined (BASALT_BUILD_CLI is OFF)";
}

#endif
