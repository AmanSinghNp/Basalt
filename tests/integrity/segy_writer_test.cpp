#include "basalt/io/segy_parser.hpp"
#include "basalt/io/segy_writer.hpp"
#include "basalt/segy_format.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define BASALT_TEST_GETPID _getpid
#else
#include <unistd.h>
#define BASALT_TEST_GETPID getpid
#endif

using namespace basalt;
using namespace basalt::io;

namespace {

using Bytes = std::vector<uint8_t>;
namespace fs = std::filesystem;

void put(Bytes& b, size_t off, uint64_t value, size_t width, bool le) {
    for (size_t i = 0; i < width; ++i) {
        const size_t shift = 8 * (le ? i : width - 1 - i);
        b[off + i] = static_cast<uint8_t>((value >> shift) & 0xFF);
    }
}

struct Spec {
    uint16_t format = 5;
    uint16_t ns = 9;
    size_t traces = 5;
    bool le = false;
    uint8_t rev_major = 1;
    size_t ext_records = 0;
    size_t trailer_records = 0;
};

// Builds a SEG-Y file with random header bytes and zero samples.
Bytes build(const Spec& s) {
    const size_t bps = segy_bytes_per_sample(s.format);
    const size_t stride = 240 + s.ns * bps;
    Bytes b(3600 + s.ext_records * 3200 + s.traces * stride + s.trailer_records * 3200, 0);

    std::mt19937 rng(1234);
    // Random text header, random ext headers, random trace headers, random trailer.
    for (size_t i = 0; i < 3200; ++i)
        b[i] = static_cast<uint8_t>(rng());
    for (size_t i = 0; i < s.ext_records * 3200; ++i)
        b[3600 + i] = static_cast<uint8_t>(rng());
    for (size_t t = 0; t < s.traces; ++t) {
        const size_t off = 3600 + s.ext_records * 3200 + t * stride;
        for (size_t i = 0; i < 240; ++i)
            b[off + i] = static_cast<uint8_t>(rng());
    }
    const size_t trailer_off = 3600 + s.ext_records * 3200 + s.traces * stride;
    for (size_t i = 0; i < s.trailer_records * 3200; ++i)
        b[trailer_off + i] = static_cast<uint8_t>(rng());

    const size_t bin = 3200;
    put(b, bin + 16, 2000, 2, s.le);
    put(b, bin + 20, s.ns, 2, s.le);
    put(b, bin + 24, s.format, 2, s.le);
    put(b, bin + 96, s.le ? 0x04030201u : (s.rev_major >= 2 ? 0x01020304u : 0u), 4, false);
    b[bin + 300] = s.rev_major;
    b[bin + 301] = 0;
    put(b, bin + 302, 1, 2, s.le);
    put(b, bin + 304, s.ext_records, 2, s.le);
    put(b, bin + 306, 0, 4, s.le);
    put(b, bin + 328, s.trailer_records, 4, s.le);
    // Clear the random bytes the binary header must not contain (unassigned
    // area 100..299 and 332..399 stay zero already; only the text header,
    // ext headers and trace headers are random).
    return b;
}

class TempDir {
public:
    TempDir() {
        static int counter = 0;
        m_path = fs::temp_directory_path() /
                 ("basalt_writer_test_" + std::to_string(static_cast<long>(BASALT_TEST_GETPID())) +
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
    [[nodiscard]] const fs::path& path() const { return m_path; }

private:
    fs::path m_path;
};

Bytes slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::vector<float> make_samples(size_t traces, size_t ns) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> v(traces * ns);
    for (float& x : v)
        x = dist(rng);
    // Include some awkward values.
    if (v.size() > 4) {
        v[0] = 0.0f;
        v[1] = 1.0f;
        v[2] = -1.0e-20f;
        v[3] = 3.0e20f;
    }
    return v;
}

size_t dir_entries(const fs::path& p) {
    return static_cast<size_t>(std::distance(fs::directory_iterator(p), fs::directory_iterator()));
}

// Writes 'src' with new samples, parses the result and checks that all header
// bytes match the source (except 'skip_off' .. 'skip_off + skip_len' of the
// file header, used for the patched format code).
Bytes round_trip(const Bytes& src, const std::vector<float>& samples, const TempDir& dir,
                 SegyLayout* out_layout = nullptr) {
    const SegyLayout in = describe_segy(src.data(), src.size());
    const std::string out_path = dir.file("out.segy");
    write_segy_like(out_path, src.data(), src.size(), in, samples.data());
    Bytes out = slurp(out_path);
    const SegyLayout lay = describe_segy(out.data(), out.size());
    EXPECT_EQ(lay.n_traces, in.n_traces);
    EXPECT_EQ(lay.samples_per_trace, in.samples_per_trace);
    EXPECT_EQ(lay.data_offset, in.data_offset);
    EXPECT_EQ(lay.little_endian, in.little_endian);
    EXPECT_EQ(lay.trailer_bytes, in.trailer_bytes);
    EXPECT_EQ(dir_entries(dir.path()), 1u) << "temporary file left behind";
    if (out_layout)
        *out_layout = lay;
    return out;
}

void expect_headers_equal(const Bytes& src, const SegyLayout& in, const Bytes& out,
                          const SegyLayout& lay, bool format_patched) {
    ASSERT_EQ(in.data_offset, lay.data_offset);
    for (size_t i = 0; i < in.data_offset; ++i) {
        if (format_patched && (i == 3224 || i == 3225))
            continue;
        ASSERT_EQ(src[i], out[i]) << "file header byte " << i;
    }
    for (size_t t = 0; t < in.n_traces; ++t) {
        const uint8_t* a = src.data() + in.data_offset + t * in.trace_stride;
        const uint8_t* b = out.data() + lay.data_offset + t * lay.trace_stride;
        ASSERT_EQ(0, std::memcmp(a, b, 240)) << "trace header " << t;
    }
    ASSERT_EQ(0, std::memcmp(src.data() + src.size() - in.trailer_bytes,
                             out.data() + out.size() - lay.trailer_bytes, in.trailer_bytes));
}

} // namespace

TEST(SegyWriterTest, IeeeBigEndianRoundTripIsExact) {
    TempDir dir;
    Spec s;
    const Bytes src = build(s);
    const SegyLayout in = describe_segy(src.data(), src.size());
    const auto samples = make_samples(s.traces, s.ns);

    SegyLayout lay;
    const Bytes out = round_trip(src, samples, dir, &lay);
    EXPECT_EQ(lay.data_sample_format, 5);
    expect_headers_equal(src, in, out, lay, false);
    for (size_t t = 0; t < s.traces; ++t) {
        const auto got = read_trace_samples(out.data(), out.size(), lay, t);
        ASSERT_EQ(got.size(), s.ns);
        for (size_t k = 0; k < s.ns; ++k) {
            EXPECT_EQ(0, std::memcmp(&got[k], &samples[t * s.ns + k], sizeof(float)))
                << "trace " << t << " sample " << k;
        }
    }
}

TEST(SegyWriterTest, IeeeLittleEndianRev2RoundTripIsExact) {
    TempDir dir;
    Spec s;
    s.le = true;
    s.rev_major = 2;
    s.ns = 6;
    s.traces = 4;
    const Bytes src = build(s);
    const SegyLayout in = describe_segy(src.data(), src.size());
    ASSERT_TRUE(in.little_endian);
    const auto samples = make_samples(s.traces, s.ns);

    SegyLayout lay;
    const Bytes out = round_trip(src, samples, dir, &lay);
    EXPECT_TRUE(lay.little_endian);
    expect_headers_equal(src, in, out, lay, false);
    // Raw check of the encoding: first sample of trace 1 is little-endian.
    const size_t off = lay.data_offset + lay.trace_stride + 240;
    uint32_t bits;
    std::memcpy(&bits, &samples[s.ns], 4);
    const uint32_t got =
        static_cast<uint32_t>(out[off]) | (static_cast<uint32_t>(out[off + 1]) << 8) |
        (static_cast<uint32_t>(out[off + 2]) << 16) | (static_cast<uint32_t>(out[off + 3]) << 24);
    EXPECT_EQ(got, bits);
    for (size_t t = 0; t < s.traces; ++t) {
        const auto back = read_trace_samples(out.data(), out.size(), lay, t);
        for (size_t k = 0; k < s.ns; ++k) {
            EXPECT_EQ(0, std::memcmp(&back[k], &samples[t * s.ns + k], sizeof(float)));
        }
    }
}

TEST(SegyWriterTest, IbmRoundTripWithinRounding) {
    for (bool le : {false, true}) {
        TempDir dir;
        Spec s;
        s.format = 1;
        s.le = le;
        s.rev_major = le ? 2 : 1;
        const Bytes src = build(s);
        const SegyLayout in = describe_segy(src.data(), src.size());
        const auto samples = make_samples(s.traces, s.ns);

        SegyLayout lay;
        const Bytes out = round_trip(src, samples, dir, &lay);
        EXPECT_EQ(lay.data_sample_format, 1);
        expect_headers_equal(src, in, out, lay, false);
        for (size_t t = 0; t < s.traces; ++t) {
            const auto got = read_trace_samples(out.data(), out.size(), lay, t);
            for (size_t k = 0; k < s.ns; ++k) {
                const float want = samples[t * s.ns + k];
                // IBM keeps 21..24 significant bits: half ulp <= 2^-21 relative.
                EXPECT_NEAR(got[k], want, std::fabs(want) * 5.0e-7f) << "t=" << t << " k=" << k;
            }
        }
    }
}

TEST(SegyWriterTest, OtherFormatsAreWrittenAsIeeeAndOnlyFormatCodeChanges) {
    for (uint16_t fmt : {uint16_t{3}, uint16_t{2}, uint16_t{6}, uint16_t{8}}) {
        for (bool le : {false, true}) {
            TempDir dir;
            Spec s;
            s.format = fmt;
            s.le = le;
            s.rev_major = le ? 2 : 1;
            const Bytes src = build(s);
            const SegyLayout in = describe_segy(src.data(), src.size());
            const auto samples = make_samples(s.traces, s.ns);

            SegyLayout lay;
            const Bytes out = round_trip(src, samples, dir, &lay);
            EXPECT_EQ(lay.data_sample_format, 5) << "format " << fmt;
            EXPECT_EQ(lay.bytes_per_sample, 4u);
            expect_headers_equal(src, in, out, lay, true);
            const auto got = read_trace_samples(out.data(), out.size(), lay, 2);
            for (size_t k = 0; k < s.ns; ++k) {
                EXPECT_EQ(got[k], samples[2 * s.ns + k]);
            }
        }
    }
}

TEST(SegyWriterTest, ExtendedTextHeadersAndTrailerArePreserved) {
    TempDir dir;
    Spec s;
    s.rev_major = 2;
    s.ext_records = 2;
    s.trailer_records = 1;
    const Bytes src = build(s);
    const SegyLayout in = describe_segy(src.data(), src.size());
    ASSERT_EQ(in.n_ext_text_headers, 2u);
    ASSERT_EQ(in.trailer_bytes, 3200u);
    const auto samples = make_samples(s.traces, s.ns);

    SegyLayout lay;
    const Bytes out = round_trip(src, samples, dir, &lay);
    EXPECT_EQ(lay.n_ext_text_headers, 2u);
    expect_headers_equal(src, in, out, lay, false);
    EXPECT_EQ(out.size(), src.size());
}

TEST(SegyWriterTest, ZeroTraceFileRoundTrips) {
    TempDir dir;
    Spec s;
    s.traces = 0;
    const Bytes src = build(s);
    const SegyLayout in = describe_segy(src.data(), src.size());
    ASSERT_EQ(in.n_traces, 0u);
    write_segy_like(dir.file("out.segy"), src.data(), src.size(), in, nullptr);
    EXPECT_EQ(slurp(dir.file("out.segy")), src);
}

TEST(SegyWriterTest, UncommittedWriterLeavesDestinationUntouchedAndNoTempFile) {
    TempDir dir;
    const Bytes src = build(Spec{});
    const SegyLayout in = describe_segy(src.data(), src.size());
    const std::string dest = dir.file("dest.segy");
    {
        std::ofstream f(dest, std::ios::binary);
        f << "previous contents";
    }
    {
        SegyWriter w(dest, src.data(), in.data_offset, in);
        const std::vector<float> trace(in.samples_per_trace, 1.0f);
        w.write_trace(src.data() + in.data_offset, trace.data());
        // destroyed without commit()
    }
    EXPECT_EQ(slurp(dest), Bytes({'p', 'r', 'e', 'v', 'i', 'o', 'u', 's', ' ', 'c', 'o', 'n', 't',
                                  'e', 'n', 't', 's'}));
    EXPECT_EQ(dir_entries(dir.path()), 1u);
}

TEST(SegyWriterTest, CommitWithWrongTraceCountThrowsAndCleansUp) {
    TempDir dir;
    const Bytes src = build(Spec{});
    const SegyLayout in = describe_segy(src.data(), src.size());
    const std::string dest = dir.file("dest.segy");
    SegyWriter w(dest, src.data(), in.data_offset, in);
    const std::vector<float> trace(in.samples_per_trace, 1.0f);
    w.write_trace(src.data() + in.data_offset, trace.data());
    EXPECT_THROW(w.commit(), SegyWriteError);
    EXPECT_FALSE(fs::exists(dest));
    EXPECT_EQ(dir_entries(dir.path()), 0u);
    EXPECT_THROW(w.write_trace(src.data() + in.data_offset, trace.data()), SegyWriteError);
}

TEST(SegyWriterTest, RejectsMisuse) {
    TempDir dir;
    const Bytes src = build(Spec{});
    const SegyLayout in = describe_segy(src.data(), src.size());
    EXPECT_THROW(SegyWriter(dir.file("a.segy"), src.data(), in.data_offset - 1, in),
                 SegyWriteError);
    EXPECT_THROW(SegyWriter(dir.file("a.segy"), nullptr, in.data_offset, in), SegyWriteError);
    // Destination directory does not exist.
    EXPECT_THROW(SegyWriter(dir.file("nodir/a.segy"), src.data(), in.data_offset, in),
                 SegyWriteError);
    // Too many traces.
    SegyWriter w(dir.file("a.segy"), src.data(), in.data_offset, in);
    const std::vector<float> trace(in.samples_per_trace, 0.0f);
    for (size_t i = 0; i < in.n_traces; ++i)
        w.write_trace(src.data() + in.data_offset, trace.data());
    EXPECT_THROW(w.write_trace(src.data() + in.data_offset, trace.data()), SegyWriteError);
    // Source buffer truncated.
    EXPECT_THROW(write_segy_like(dir.file("b.segy"), src.data(), src.size() - 1, in,
                                 make_samples(in.n_traces, in.samples_per_trace).data()),
                 SegyWriteError);
}
