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
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
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
    basalt::io::write_segy(path, basalt::io::make_segy(samples, ntraces, nsamples, dt, format), format);
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
double central_attenuation_db(const std::vector<float>& filtered, const std::vector<float>& original,
                              size_t ntraces, size_t nsamples) {
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
const std::vector<std::string> kIdentityFlags = {"--vmin", "1e29", "--vmax", "1e30",
                                                 "--taper", "0", "--edge-taper", "0"};

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
    EXPECT_EQ(r.out.rfind("basalt ", 0), 0u) << r.out;
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
    EXPECT_TRUE(contains(info.err, "usage"));

    const CliResult filter = run_cli({"filter", in, test_path("out.segy"), "--nope"});
    EXPECT_EQ(filter.exit_code, 2);
    EXPECT_TRUE(contains(filter.err, "usage"));
}

TEST(CliUsage, BadNumberExitsTwo) {
    const std::string in = test_path("in.segy");
    write_fixture(in, random_samples(20 * 100, 2), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, test_path("out.segy"), "--vmin", "1x"});
    EXPECT_EQ(r.exit_code, 2);
    EXPECT_TRUE(contains(r.err, "usage"));
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

    const CliResult r = run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
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
    constexpr size_t kBlock = 7;  // blocks of 7, 7 and 6 traces
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    const std::string out_whole = test_path("whole.segy");
    write_fixture(in, random_samples(kTraces * kSamples, 9), kTraces, kSamples, 0.004,
                  SampleFormat::IEEE32);
    const SegyData input = basalt::io::read_segy(in);

    const CliResult blocked =
        run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5", "--gather-size", "7"});
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
        const auto begin = input.samples.begin() + static_cast<std::ptrdiff_t>(first * kSamples);
        const std::vector<float> block(begin, begin + static_cast<std::ptrdiff_t>(count * kSamples));
        const std::vector<float> filtered = basalt::fk_filter_gather(block, count, kSamples, params);
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
}

TEST(CliFilter, DtOverrideIsRecordedInOutput) {
    const std::string in = test_path("in.segy");
    const std::string out = test_path("out.segy");
    write_fixture(in, random_samples(20 * 100, 12), 20, 100, 0.004, SampleFormat::IEEE32);

    const CliResult r = run_cli({"filter", in, out, "--dt", "0.002", "--vmin", "500", "--vmax", "2000"});
    ASSERT_EQ(r.exit_code, 0) << r.err;
    EXPECT_NEAR(basalt::io::read_segy(out).dt, 0.002, 1e-12);
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

    const CliResult r = run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
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

    const CliResult r = run_cli({"filter", in, out, "--vmin", "500", "--vmax", "2000", "--dx", "5"});
    ASSERT_EQ(r.exit_code, 0) << r.err;

    const std::vector<float> filtered = basalt::io::read_segy(out).samples;
    EXPECT_GT(central_attenuation_db(filtered, event, kTraces, kSamples), -1.0);
}
