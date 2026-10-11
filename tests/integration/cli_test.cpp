// Integration tests for the basalt command-line tool. Each test runs the built binary
// (BASALT_CLI_PATH) through the shell and checks its exit status, its output streams and the
// SEG-Y files it writes. Fixtures are created under ::testing::TempDir().
// Contract: docs/design/FK_PIPELINE.md, sections 5 and 6.

#include <gtest/gtest.h>

#include "basalt/fk.hpp"
#include "basalt/io/segy_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace {

using basalt::io::SampleFormat;
using basalt::io::SegyData;

constexpr double kPi = 3.14159265358979323846;

// Exit status and captured output of one CLI run.
struct CliResult {
    int exit_code = -1;
    std::string out;
    std::string err;
};

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// Quotes one argument for the shell so that paths with spaces or quotes survive intact.
std::string shell_quote(const std::string& arg) {
#ifdef _WIN32
    return "\"" + arg + "\"";
#else
    std::string quoted = "'";
    for (const char c : arg) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    return quoted + "'";
#endif
}

// A file name unique to the running test. ctest may run tests in parallel, so tests must
// never share fixture or output paths.
std::string test_path(const std::string& suffix) {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    return ::testing::TempDir() + info->test_suite_name() + "_" + info->name() + "_" + suffix;
}

// Runs the CLI with `args` (each one quoted for the shell) and captures stdout and stderr.
CliResult run_cli(const std::vector<std::string>& args) {
    const std::string out_file = test_path("stdout.txt");
    const std::string err_file = test_path("stderr.txt");
    std::string command = shell_quote(BASALT_CLI_PATH);
    for (const std::string& arg : args) {
        command += " " + shell_quote(arg);
    }
    command += " > " + shell_quote(out_file) + " 2> " + shell_quote(err_file);

    const int status = std::system(command.c_str());
#ifdef _WIN32
    const int exit_code = status;
#else
    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    return {exit_code, read_text(out_file), read_text(err_file)};
}

void write_fixture(const std::string& path, const std::vector<float>& samples, size_t ntraces,
                   size_t nsamples, double dt, SampleFormat format) {
    basalt::io::write_segy(path, basalt::io::make_segy(samples, ntraces, nsamples, dt, format),
                           format);
}

// Writes an INT8, INT16 or INT32 SEG-Y file. The library writer produces only IBM32 and IEEE32,
// so the file is laid out here: the headers of a minimal IEEE32 file with the format code
// replaced (binary header bytes 3225-3226), then big-endian integer samples.
void write_integer_fixture(const std::string& path, size_t ntraces, size_t nsamples, double dt,
                           SampleFormat format) {
    const size_t bytes = format == SampleFormat::INT8 ? 1 : format == SampleFormat::INT16 ? 2 : 4;
    const std::vector<float> zeros(ntraces * nsamples, 0.0f);
    SegyData data = basalt::io::make_segy(zeros, ntraces, nsamples, dt, SampleFormat::IEEE32);
    const auto code = static_cast<uint16_t>(format);
    data.raw_binary_header[24] = static_cast<uint8_t>(code >> 8);
    data.raw_binary_header[25] = static_cast<uint8_t>(code & 0xFFu);

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.raw_text_header.data()),
              static_cast<std::streamsize>(data.raw_text_header.size()));
    out.write(reinterpret_cast<const char*>(data.raw_binary_header.data()),
              static_cast<std::streamsize>(data.raw_binary_header.size()));
    for (size_t tr = 0; tr < ntraces; ++tr) {
        out.write(reinterpret_cast<const char*>(data.raw_trace_headers[tr].data()),
                  static_cast<std::streamsize>(data.raw_trace_headers[tr].size()));
        for (size_t s = 0; s < nsamples; ++s) {
            // Small values in [-100, 100] fit every integer format.
            const int32_t value = static_cast<int32_t>(((tr * nsamples + s) * 37) % 201) - 100;
            const auto bits = static_cast<uint32_t>(value);
            for (size_t b = bytes; b-- > 0;) {
                out.put(static_cast<char>((bits >> (8 * b)) & 0xFFu));
            }
        }
    }
}

std::vector<float> random_samples(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> samples(count);
    for (float& s : samples) {
        s = dist(rng);
    }
    return samples;
}

// A 25 Hz Ricker event on the line t(x) = t_mid + (x - x_mid) / velocity, multiplied by a Hann
// envelope along x so that the event is spatially smooth. A positive velocity means the arrival
// time increases with trace position (docs/design/FK_PIPELINE.md, section 2).
std::vector<float> dipping_event(size_t ntraces, size_t nsamples, double dt, double dx,
                                 double velocity) {
    constexpr double kPeakHz = 25.0;
    const double t_mid = 0.5 * static_cast<double>(nsamples) * dt;
    const double x_mid = 0.5 * static_cast<double>(ntraces - 1) * dx;
    std::vector<float> samples(ntraces * nsamples, 0.0f);
    for (size_t tr = 0; tr < ntraces; ++tr) {
        const double x = static_cast<double>(tr) * dx;
        const double arrival = t_mid + (x - x_mid) / velocity;
        const double phase = 2.0 * kPi * static_cast<double>(tr) / static_cast<double>(ntraces - 1);
        const double envelope = 0.5 - 0.5 * std::cos(phase);
        for (size_t s = 0; s < nsamples; ++s) {
            const double a = kPi * kPeakHz * (static_cast<double>(s) * dt - arrival);
            const double ricker = (1.0 - 2.0 * a * a) * std::exp(-a * a);
            samples[tr * nsamples + s] = static_cast<float>(envelope * ricker);
        }
    }
    return samples;
}

// Energy of `filtered` relative to `original` over the central half of the traces, in dB.
double central_attenuation_db(const std::vector<float>& filtered,
                              const std::vector<float>& original, size_t ntraces, size_t nsamples) {
    double energy_out = 0.0;
    double energy_in = 0.0;
    for (size_t tr = ntraces / 4; tr < 3 * ntraces / 4; ++tr) {
        for (size_t s = 0; s < nsamples; ++s) {
            const size_t i = tr * nsamples + s;
            energy_out += static_cast<double>(filtered[i]) * static_cast<double>(filtered[i]);
            energy_in += static_cast<double>(original[i]) * static_cast<double>(original[i]);
        }
    }
    return 10.0 * std::log10(energy_out / energy_in);
}

// RMS of (actual - expected) divided by the RMS of expected. Sizes must match.
double relative_rms(const std::vector<float>& actual, const std::vector<float>& expected) {
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double d = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        err += d * d;
        ref += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
    }
    return std::sqrt(err / ref);
}

// Flags that make the pipeline an identity: a band no bin falls in, hard edges, no edge tapers.
const std::vector<std::string> kIdentityFlags = {"--vmin",  "1e29", "--vmax",       "1e30",
                                                 "--taper", "0",    "--edge-taper", "0"};

// Runs `filter in out` with `flags`, expects success and returns the samples written to `out`.
std::vector<float> cli_filter(const std::string& in, const std::string& out,
                              const std::vector<std::string>& flags) {
    std::vector<std::string> args = {"filter", in, out};
    args.insert(args.end(), flags.begin(), flags.end());
    const CliResult r = run_cli(args);
    EXPECT_EQ(r.exit_code, 0) << r.err;
    return basalt::io::read_segy(out).samples;
}

} // namespace

// --- Command line parsing and exit codes ---------------------------------------------------

TEST(CliUsage, HelpExitsZero) {
    const CliResult r = run_cli({"--help"});
    EXPECT_EQ(r.exit_code, 0);
    EXPECT_TRUE(contains(r.out, "usage"));
}

TEST(CliUsage, VersionPrintsVersion) {
    const CliResult r = run_cli({"--version"});
    EXPECT_EQ(r.exit_code, 0);
    EXPECT_EQ(r.out, std::string("basalt ") + BASALT_VERSION + "\n");
}

TEST(CliUsage, NoArgumentsExitsTwo) {
    const CliResult r = run_cli({});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "usage"));
}

TEST(CliUsage, UnknownCommandExitsTwo) {
    const CliResult r = run_cli({"frobnicate"});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "usage"));
}

TEST(CliUsage, UnknownFlagExitsTwo) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 1), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult info = run_cli({"info", in, "--bogus"});
    EXPECT_EQ(info.exit_code, 2);
    EXPECT_TRUE(contains(info.err, "unknown option '--bogus'")) << info.err;
    EXPECT_TRUE(contains(info.err, "usage"));

    const CliResult filter = run_cli({"filter", in, test_path("out.segy"), "--nope"});
    EXPECT_EQ(filter.exit_code, 2);
    EXPECT_TRUE(contains(filter.err, "unknown option '--nope'")) << filter.err;
    EXPECT_TRUE(contains(filter.err, "usage"));
}

TEST(CliUsage, BadNumberExitsTwo) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 2), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, test_path("out.segy"), "--vmin", "1x"});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "usage"));
}

TEST(CliUsage, HexFloatIsRejectedAsNumber) {
    // strtod would read 0x1p3 as 8; the CLI accepts decimal notation only.
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 13), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, test_path("out.segy"), "--vmin", "0x1p3"});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "invalid number '0x1p3' for --vmin")) << r.err;
}

TEST(CliUsage, MissingValueExitsTwo) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 3), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, test_path("out.segy"), "--vmin"});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "usage"));
}

TEST(CliUsage, OutOfRangeTaperExitsTwo) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 4), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, test_path("out.segy"), "--taper", "1.5"});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "--taper must be in [0, 1)")) << r.err;
}

TEST(CliUsage, SubnormalNumberIsRejectedByName) {
    // Flush-to-zero would turn 1e-320 into zero, so the value is refused with its own message.
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 16), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, test_path("out.segy"), "--dx", "1e-320"});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "too small for --dx")) << r.err;
}

TEST(CliUsage, InvalidFilterValuesExitTwoWithMessage) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 23), 20, 100, 0.004, SampleFormat::IEEE32);

    const std::vector<std::pair<std::vector<std::string>, std::string>> cases = {
        {{"--vmin", "500", "--vmax", "500"}, "--vmin must be less than --vmax"},
        {{"--pad", "0.5"}, "--pad must be at least 1"},
        {{"--dx", "0"}, "--dx must be positive"},
        {{"--gather-size", "0"}, "--gather-size must be at least 1"},
        {{"--dt", "0.0040000001"}, "--dt must be a whole number of microseconds"},
        {{"--dt", "0.07"}, "--dt must be a whole number of microseconds"},
        {{"--mode", "bogus"}, "--mode expects 'reject' or 'pass'"},
        {{"--format", "bogus"}, "--format expects 'ibm' or 'ieee'"},
        {{"--threads", "-1"}, "--threads expects a non-negative integer"},
        {{"--edge-taper", "-2"}, "--edge-taper expects a non-negative integer"},
    };
    for (const auto& [flags, message] : cases) {
        std::vector<std::string> args = {"filter", in, test_path("out.segy")};
        args.insert(args.end(), flags.begin(), flags.end());
        const CliResult r = run_cli(args);
        EXPECT_EQ(r.exit_code, 2) << flags[0] << " " << flags[1];
        EXPECT_TRUE(contains(r.err, message)) << r.err;
        EXPECT_TRUE(contains(r.err, "usage")) << r.err;
    }
}

TEST(CliUsage, SameInputAndOutputIsRejectedAndInputUntouched) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 17), 20, 100, 0.004, SampleFormat::IEEE32);
    const std::string before = read_text(in);

    const CliResult r = run_cli({"filter", in, in});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "must be different files")) << r.err;
    EXPECT_EQ(read_text(in), before);
}

TEST(CliRuntime, MissingInputFileExitsOne) {
    const std::string missing = test_path("no_such_file.segy");

    const CliResult info = run_cli({"info", missing});
    EXPECT_EQ(info.exit_code, 1);
    EXPECT_TRUE(contains(info.err, "cannot open"));

    const CliResult filter = run_cli({"filter", missing, test_path("out.segy")});
    EXPECT_EQ(filter.exit_code, 1);
    EXPECT_TRUE(contains(filter.err, "cannot open"));
}

TEST(CliRuntime, ZeroTraceInputIsRejectedWithoutOutput) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    write_fixture(in, {}, 0, 100, 0.004, SampleFormat::IEEE32);
    std::remove(out.c_str());

    const CliResult info = run_cli({"info", in});
    EXPECT_EQ(info.exit_code, 0) << info.err;
    EXPECT_TRUE(contains(info.out, "ntraces: 0")) << info.out;

    const CliResult filter = run_cli({"filter", in, out});
    EXPECT_EQ(filter.exit_code, 1);
    EXPECT_TRUE(contains(filter.err, "no traces")) << filter.err;
    EXPECT_FALSE(std::ifstream(out).good());
}

TEST(CliRuntime, ZeroHeaderIntervalIsNotRepairedByDt) {
    // The reader rejects a binary header whose sample interval is zero, before --dt is applied.
    // Bytes 3217-3218 (1-based) hold the interval, so they sit at offset 3216.
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    write_fixture(in, random_samples(20 * 100, 14), 20, 100, 0.004, SampleFormat::IEEE32);
    {
        std::fstream file(in, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(3216);
        const char zeros[2] = {0, 0};
        file.write(zeros, 2);
    }

    const CliResult r = run_cli({"filter", in, out, "--dt", "0.002"});
    EXPECT_EQ(r.exit_code, 1);
    EXPECT_TRUE(contains(r.err, "zero sample interval")) << r.err;
}

TEST(CliRuntime, OversizedPadReportsOutOfMemory) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 15), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, test_path("out.segy"), "--pad", "1e6"});
    EXPECT_EQ(r.exit_code, 1);
    EXPECT_TRUE(contains(r.err, "out of memory")) << r.err;
    EXPECT_FALSE(contains(r.err, "bad_alloc")) << r.err;
}

// --- info ----------------------------------------------------------------------------------

TEST(CliInfo, PrintsGeometryAndTextHeader) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 5), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"info", in});
    EXPECT_EQ(r.exit_code, 0) << r.err;
    EXPECT_TRUE(contains(r.out, "ntraces: 20")) << r.out;
    EXPECT_TRUE(contains(r.out, "nsamples: 100")) << r.out;
    EXPECT_TRUE(contains(r.out, "dt: 0.004 s (4000 us)")) << r.out;
    EXPECT_TRUE(contains(r.out, "format: IEEE32")) << r.out;
    EXPECT_TRUE(contains(r.out, "C 1 CREATED BY BASALT")) << r.out;
}

// --- filter --------------------------------------------------------------------------------

TEST(CliFilter, OutputHasSameGeometryAsInput) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    write_fixture(in, random_samples(20 * 100, 6), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r =
        run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
    ASSERT_EQ(r.exit_code, 0) << r.err;
    EXPECT_TRUE(contains(r.out, "filtered 20 traces, 100 samples")) << r.out;

    const SegyData result = basalt::io::read_segy(out);
    EXPECT_EQ(result.ntraces, std::size_t{20});
    EXPECT_EQ(result.nsamples, std::size_t{100});
    EXPECT_NEAR(result.dt, 0.004, 1e-12);
    EXPECT_EQ(result.samples.size(), std::size_t{20 * 100});
}

TEST(CliFilter, IdentityConfigurationReproducesInput) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> samples = random_samples(20 * 100, 7);
    write_fixture(in, samples, 20, 100, 0.004, SampleFormat::IEEE32);

    std::vector<std::string> args = {"filter", in, out};
    args.insert(args.end(), kIdentityFlags.begin(), kIdentityFlags.end());
    const CliResult r = run_cli(args);
    ASSERT_EQ(r.exit_code, 0) << r.err;

    const std::vector<float> result = basalt::io::read_segy(out).samples;
    ASSERT_EQ(result.size(), samples.size());
    EXPECT_LT(relative_rms(result, samples), 1e-5);
}

TEST(CliFilter, GatherSizeOneWorksAndIsIdentityForIdentityConfiguration) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> samples = random_samples(5 * 64, 8);
    write_fixture(in, samples, 5, 64, 0.004, SampleFormat::IEEE32);

    std::vector<std::string> args = {"filter", in, out, "--gather-size", "1"};
    args.insert(args.end(), kIdentityFlags.begin(), kIdentityFlags.end());
    const CliResult r = run_cli(args);
    ASSERT_EQ(r.exit_code, 0) << r.err;

    const SegyData result = basalt::io::read_segy(out);
    EXPECT_EQ(result.ntraces, std::size_t{5});
    ASSERT_EQ(result.samples.size(), samples.size());
    EXPECT_LT(relative_rms(result.samples, samples), 1e-5);
}

TEST(CliFilter, GatherSizeMatchesFilteringEachBlockWithLibrary) {
    constexpr size_t kTraces = 20;
    constexpr size_t kSamples = 100;
    constexpr size_t kBlock = 7; // blocks of 7, 7 and 6 traces
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::string out_whole = test_path("whole.segy");
    write_fixture(in, random_samples(kTraces * kSamples, 9), kTraces, kSamples, 0.004,
                  SampleFormat::IEEE32);
    const SegyData input = basalt::io::read_segy(in);

    const CliResult blocked = run_cli(
        {"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5", "--gather-size", "7"});
    ASSERT_EQ(blocked.exit_code, 0) << blocked.err;
    const std::vector<float> cli_result = basalt::io::read_segy(out).samples;

    basalt::FKMuteParams params;
    params.dt = input.dt;
    params.dx = 5.0;
    params.v_min = 500.0;
    params.v_max = 2000.0;
    std::vector<float> expected;
    for (size_t first = 0; first < kTraces; first += kBlock) {
        const size_t count = std::min(kBlock, kTraces - first);
        // Proposed per-block rule, pending a contract decision (not yet in FK_PIPELINE.md): the
        // CLI reduces the default edge taper of 4 to (count - 1) / 2 for each block. If the rule
        // is rejected, this expectation and the reduction in src/main.cpp revert together.
        params.edge_taper_traces = std::min<size_t>(4, (count - 1) / 2);
        const auto begin = input.samples.begin() + static_cast<std::ptrdiff_t>(first * kSamples);
        const std::vector<float> block(begin,
                                       begin + static_cast<std::ptrdiff_t>(count * kSamples));
        const std::vector<float> filtered =
            basalt::fk_filter_gather(block, count, kSamples, params);
        expected.insert(expected.end(), filtered.begin(), filtered.end());
    }
    ASSERT_EQ(cli_result.size(), expected.size());
    EXPECT_LT(relative_rms(cli_result, expected), 1e-6);

    // The blocks are real: filtering the whole 20-trace gather gives a different result.
    const CliResult whole =
        run_cli({"filter", in, out_whole, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
    ASSERT_EQ(whole.exit_code, 0) << whole.err;
    EXPECT_GT(relative_rms(basalt::io::read_segy(out_whole).samples, expected), 1e-3);
}

TEST(CliFilter, IbmInputKeepsIbmOutputAndFormatOverrides) {
    const std::string in_ibm = test_path("in_ibm.segy");
    const std::string in_ieee = test_path("in_ieee.segy");
    const std::string out_default = test_path("default.segy");
    const std::string out_ieee = test_path("ieee.segy");
    const std::string out_ibm = test_path("ibm.segy");
    write_fixture(in_ibm, random_samples(20 * 100, 10), 20, 100, 0.004, SampleFormat::IBM32);
    write_fixture(in_ieee, random_samples(20 * 100, 11), 20, 100, 0.004, SampleFormat::IEEE32);
    const std::vector<std::string> filter_flags = {"--vmin", "500", "--vmax", "2000", "--dx", "5"};

    std::vector<std::string> args = {"filter", in_ibm, out_default};
    args.insert(args.end(), filter_flags.begin(), filter_flags.end());
    ASSERT_EQ(run_cli(args).exit_code, 0);
    EXPECT_EQ(static_cast<int>(basalt::io::read_segy(out_default).format),
              static_cast<int>(SampleFormat::IBM32));

    args = {"filter", in_ibm, out_ieee, "--format", "ieee"};
    args.insert(args.end(), filter_flags.begin(), filter_flags.end());
    ASSERT_EQ(run_cli(args).exit_code, 0);
    EXPECT_EQ(static_cast<int>(basalt::io::read_segy(out_ieee).format),
              static_cast<int>(SampleFormat::IEEE32));

    args = {"filter", in_ieee, out_ibm, "--format", "ibm"};
    args.insert(args.end(), filter_flags.begin(), filter_flags.end());
    ASSERT_EQ(run_cli(args).exit_code, 0);
    EXPECT_EQ(static_cast<int>(basalt::io::read_segy(out_ibm).format),
              static_cast<int>(SampleFormat::IBM32));

    // The format code is not enough: each output, decoded by the reader, must hold the filtered
    // samples. The expected values come from the library run on the decoded input. IBM32 keeps
    // about seven significant digits, so its outputs are compared with a looser bound.
    basalt::FKMuteParams params;
    params.dx = 5.0;
    params.v_min = 500.0;
    params.v_max = 2000.0;
    const SegyData ibm_input = basalt::io::read_segy(in_ibm);
    const SegyData ieee_input = basalt::io::read_segy(in_ieee);
    params.dt = ibm_input.dt;
    const std::vector<float> expected_ibm =
        basalt::fk_filter_gather(ibm_input.samples, 20, 100, params);
    params.dt = ieee_input.dt;
    const std::vector<float> expected_ieee =
        basalt::fk_filter_gather(ieee_input.samples, 20, 100, params);
    EXPECT_LT(relative_rms(basalt::io::read_segy(out_default).samples, expected_ibm), 1e-5);
    EXPECT_LT(relative_rms(basalt::io::read_segy(out_ieee).samples, expected_ibm), 1e-6);
    EXPECT_LT(relative_rms(basalt::io::read_segy(out_ibm).samples, expected_ieee), 1e-5);
}

TEST(CliFilter, DtOverrideIsRecordedInOutput) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    write_fixture(in, random_samples(20 * 100, 12), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r =
        run_cli({"filter", in, out, "--dt", "0.002", "--vmin", "500", "--vmax", "2000"});
    ASSERT_EQ(r.exit_code, 0) << r.err;
    EXPECT_NEAR(basalt::io::read_segy(out).dt, 0.002, 1e-12);
}

TEST(CliFilter, DtOverrideChangesTheVelocityMapping) {
    constexpr size_t kTraces = 64;
    constexpr size_t kSamples = 500;
    constexpr double kDt = 0.002;
    constexpr double kDx = 5.0;
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    // The event is sampled every 2 ms, but the header claims 4 ms. Apparent velocity is
    // v = -f / k, so the header interval halves every velocity: +1000 m/s would read as +500.
    const std::vector<float> event = dipping_event(kTraces, kSamples, kDt, kDx, +1000.0);
    write_fixture(in, event, kTraces, kSamples, 0.004, SampleFormat::IEEE32);

    const std::vector<float> with_dt =
        cli_filter(in, out, {"--dt", "0.002", "--vmin", "500", "--vmax", "2000", "--dx", "5"});
    basalt::FKMuteParams params;
    params.dt = kDt;
    params.dx = kDx;
    params.v_min = 500.0;
    params.v_max = 2000.0;
    const std::vector<float> expected = basalt::fk_filter_gather(event, kTraces, kSamples, params);
    ASSERT_EQ(with_dt.size(), expected.size());
    EXPECT_LT(relative_rms(with_dt, expected), 1e-6);
    EXPECT_LE(central_attenuation_db(with_dt, event, kTraces, kSamples), -20.0);

    // Without --dt the header interval is used, and the same band gives a different filter.
    const std::vector<float> header_dt =
        cli_filter(in, out, {"--vmin", "500", "--vmax", "2000", "--dx", "5"});
    ASSERT_EQ(header_dt.size(), expected.size());
    EXPECT_GT(relative_rms(header_dt, expected), 1e-2);
}

TEST(CliFilter, AttenuatesPositiveVelocityEvent) {
    constexpr size_t kTraces = 64;
    constexpr size_t kSamples = 500;
    constexpr double kDt = 0.002;
    constexpr double kDx = 5.0;
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> event = dipping_event(kTraces, kSamples, kDt, kDx, +1000.0);
    write_fixture(in, event, kTraces, kSamples, kDt, SampleFormat::IEEE32);

    const CliResult r =
        run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
    ASSERT_EQ(r.exit_code, 0) << r.err;

    const std::vector<float> filtered = basalt::io::read_segy(out).samples;
    EXPECT_LE(central_attenuation_db(filtered, event, kTraces, kSamples), -20.0);
}

TEST(CliFilter, KeepsNegativeVelocityEvent) {
    constexpr size_t kTraces = 64;
    constexpr size_t kSamples = 500;
    constexpr double kDt = 0.002;
    constexpr double kDx = 5.0;
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> event = dipping_event(kTraces, kSamples, kDt, kDx, -1000.0);
    write_fixture(in, event, kTraces, kSamples, kDt, SampleFormat::IEEE32);

    const CliResult r =
        run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
    ASSERT_EQ(r.exit_code, 0) << r.err;

    const std::vector<float> filtered = basalt::io::read_segy(out).samples;
    EXPECT_GT(central_attenuation_db(filtered, event, kTraces, kSamples), -1.0);
}

TEST(CliFilter, PassModeKeepsBandEventAndRemovesOtherEvent) {
    constexpr size_t kTraces = 64;
    constexpr size_t kSamples = 500;
    constexpr double kDt = 0.002;
    constexpr double kDx = 5.0;
    const std::string in_pos = test_path("in_pos.segy");
    const std::string in_neg = test_path("in_neg.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> pos = dipping_event(kTraces, kSamples, kDt, kDx, +1000.0);
    const std::vector<float> neg = dipping_event(kTraces, kSamples, kDt, kDx, -1000.0);
    write_fixture(in_pos, pos, kTraces, kSamples, kDt, SampleFormat::IEEE32);
    write_fixture(in_neg, neg, kTraces, kSamples, kDt, SampleFormat::IEEE32);
    const std::vector<std::string> flags = {"--mode", "pass", "--vmin", "500",
                                            "--vmax", "2000", "--dx",   "5"};

    // The band [500, 2000] m/s keeps the +1000 event and removes the -1000 event.
    const std::vector<float> kept = cli_filter(in_pos, out, flags);
    EXPECT_GT(central_attenuation_db(kept, pos, kTraces, kSamples), -1.0);
    const std::vector<float> removed = cli_filter(in_neg, out, flags);
    EXPECT_LE(central_attenuation_db(removed, neg, kTraces, kSamples), -20.0);
}

TEST(CliFilter, DefaultParametersMatchLibraryDefaults) {
    constexpr size_t kTraces = 40;
    constexpr size_t kSamples = 200;
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> samples = random_samples(kTraces * kSamples, 18);
    write_fixture(in, samples, kTraces, kSamples, 0.004, SampleFormat::IEEE32);
    const SegyData input = basalt::io::read_segy(in);

    // Only the band is given. The rest must take the defaults: dx 25 m, taper 0.1, reject mode,
    // pad 1.5, edge taper of 4 traces, one thread.
    const std::vector<float> cli_result = cli_filter(in, out, {"--vmin", "500", "--vmax", "2000"});

    basalt::FKMuteParams params;
    params.dt = input.dt;
    params.v_min = 500.0;
    params.v_max = 2000.0;
    const std::vector<float> expected =
        basalt::fk_filter_gather(samples, kTraces, kSamples, params);
    EXPECT_LT(relative_rms(cli_result, expected), 1e-6);

    // A 5 m trace spacing gives a clearly different mute, so the default dx is really checked.
    params.dx = 5.0;
    const std::vector<float> other_dx =
        basalt::fk_filter_gather(samples, kTraces, kSamples, params);
    EXPECT_GT(relative_rms(cli_result, other_dx), 1e-2);
}

TEST(CliFilter, PadEdgeTaperAndThreadsReachTheLibrary) {
    constexpr size_t kTraces = 30;
    constexpr size_t kSamples = 150;
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> samples = random_samples(kTraces * kSamples, 19);
    write_fixture(in, samples, kTraces, kSamples, 0.004, SampleFormat::IEEE32);

    basalt::FKMuteParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.v_min = 500.0;
    params.v_max = 2000.0;
    params.pad_factor = 2.0;
    params.edge_taper_traces = 2;
    // Threads change how the FFT is computed, not its result, so the library runs single-threaded.
    const std::vector<float> expected =
        basalt::fk_filter_gather(samples, kTraces, kSamples, params);
    const std::vector<float> cli_result =
        cli_filter(in, out,
                   {"--vmin", "500", "--vmax", "2000", "--dx", "5", "--pad", "2.0", "--edge-taper",
                    "2", "--threads", "3"});
    EXPECT_LT(relative_rms(cli_result, expected), 1e-6);

    // The flags matter: the library defaults for pad and edge taper give a different result.
    basalt::FKMuteParams defaults = params;
    defaults.pad_factor = 1.5;
    defaults.edge_taper_traces = 4;
    EXPECT_GT(
        relative_rms(basalt::fk_filter_gather(samples, kTraces, kSamples, defaults), expected),
        1e-3);
}

TEST(CliFilter, EdgeTaperIsReducedWithWarningForShortBlocks) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    write_fixture(in, random_samples(20 * 100, 20), 20, 100, 0.004, SampleFormat::IEEE32);

    // Default taper 4 on blocks of 7 and a last block of 6 traces: at most 3 and 2 traces per end
    // fit, so both sizes are reduced and each gets its own warning.
    const CliResult tapered = run_cli({"filter", in, out, "--gather-size", "7"});
    EXPECT_EQ(tapered.exit_code, 0) << tapered.err;
    EXPECT_TRUE(
        contains(tapered.err, "warning: --edge-taper 4 is reduced to 3 per end for 7-trace blocks"))
        << tapered.err;
    EXPECT_TRUE(contains(
        tapered.err, "warning: --edge-taper 4 is reduced to 2 per end for the last 6-trace block"))
        << tapered.err;

    // Blocks of 10 traces fit a taper of 4 per end, and no taper means no warning.
    const CliResult long_blocks = run_cli({"filter", in, out, "--gather-size", "10"});
    EXPECT_EQ(long_blocks.exit_code, 0) << long_blocks.err;
    EXPECT_FALSE(contains(long_blocks.err, "warning")) << long_blocks.err;

    const CliResult no_taper =
        run_cli({"filter", in, out, "--gather-size", "7", "--edge-taper", "0"});
    EXPECT_EQ(no_taper.exit_code, 0) << no_taper.err;
    EXPECT_FALSE(contains(no_taper.err, "warning")) << no_taper.err;
}

TEST(CliFilter, WarningNamesOnlyTheShortTailBlockWhenOnlyItIsReduced) {
    // Blocks of 10 traces keep the default taper of 4, but the last block of 2 traces gets none.
    // The warning must name the tail block and must not claim that the 10-trace blocks change.
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    write_fixture(in, random_samples(22 * 100, 24), 22, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, out, "--gather-size", "10"});
    EXPECT_EQ(r.exit_code, 0) << r.err;
    EXPECT_TRUE(contains(
        r.err, "warning: --edge-taper 4 is reduced to 0 per end for the last 2-trace block"))
        << r.err;
    EXPECT_FALSE(contains(r.err, "10-trace blocks")) << r.err;
}

TEST(CliFilter, IntegerInputIsWrittenAsIeee32ByDefault) {
    constexpr size_t kTraces = 20;
    constexpr size_t kSamples = 100;
    const std::vector<std::pair<SampleFormat, std::string>> cases = {
        {SampleFormat::INT8, "INT8"},
        {SampleFormat::INT16, "INT16"},
        {SampleFormat::INT32, "INT32"}};
    for (const auto& [format, name] : cases) {
        const std::string in = test_path(name + "_in.segy");
        const std::string out = test_path(name + "_out.segy");
        write_integer_fixture(in, kTraces, kSamples, 0.004, format);

        const CliResult info = run_cli({"info", in});
        ASSERT_EQ(info.exit_code, 0) << info.err;
        EXPECT_TRUE(contains(info.out, "format: " + name + " ")) << info.out;

        const CliResult r =
            run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
        ASSERT_EQ(r.exit_code, 0) << r.err;
        EXPECT_TRUE(contains(r.out, "(IEEE32)")) << r.out;

        // The output is IEEE32 and holds the library result for the integer values as read.
        const SegyData result = basalt::io::read_segy(out);
        EXPECT_EQ(static_cast<int>(result.format), static_cast<int>(SampleFormat::IEEE32)) << name;
        basalt::FKMuteParams params;
        params.dt = 0.004;
        params.dx = 5.0;
        params.v_min = 500.0;
        params.v_max = 2000.0;
        const std::vector<float> decoded = basalt::io::read_segy(in).samples;
        const std::vector<float> expected =
            basalt::fk_filter_gather(decoded, kTraces, kSamples, params);
        ASSERT_EQ(result.samples.size(), expected.size());
        EXPECT_LT(relative_rms(result.samples, expected), 1e-6) << name;
    }
}

TEST(CliFilter, SingleTraceAndSingleSampleFilesReproduceInputInIdentityConfiguration) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");

    const std::vector<float> one_trace = random_samples(3, 21);
    write_fixture(in, one_trace, 1, 3, 0.004, SampleFormat::IEEE32);
    const std::vector<float> trace_result = cli_filter(in, out, kIdentityFlags);
    ASSERT_EQ(trace_result.size(), one_trace.size());
    EXPECT_LT(relative_rms(trace_result, one_trace), 1e-5);

    const std::vector<float> one_sample = random_samples(5, 22);
    write_fixture(in, one_sample, 5, 1, 0.004, SampleFormat::IEEE32);
    const std::vector<float> sample_result = cli_filter(in, out, kIdentityFlags);
    ASSERT_EQ(sample_result.size(), one_sample.size());
    EXPECT_LT(relative_rms(sample_result, one_sample), 1e-5);
}

TEST(CliFilter, SingleTraceIsNotTaperedByDefault) {
    constexpr size_t kSamples = 50;
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::vector<float> samples = random_samples(kSamples, 23);
    write_fixture(in, samples, 1, kSamples, 0.004, SampleFormat::IEEE32);

    // A one-trace block has no room for an edge taper, so the default taper of 4 is reduced to
    // zero and the CLI says so. The output equals the library run without edge tapers.
    const CliResult r = run_cli({"filter", in, out});
    ASSERT_EQ(r.exit_code, 0) << r.err;
    EXPECT_TRUE(
        contains(r.err, "warning: --edge-taper 4 is reduced to 0 per end for 1-trace blocks"))
        << r.err;

    basalt::FKMuteParams params;
    params.dt = 0.004;
    params.edge_taper_traces = 0;
    const std::vector<float> expected = basalt::fk_filter_gather(samples, 1, kSamples, params);
    const std::vector<float> result = basalt::io::read_segy(out).samples;
    ASSERT_EQ(result.size(), expected.size());
    EXPECT_LT(relative_rms(result, expected), 1e-6);
}
