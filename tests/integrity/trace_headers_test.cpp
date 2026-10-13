#include "basalt/io/segy_file.hpp"
#include "basalt/io/trace_headers.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basalt::io;

namespace {

using RawHeader = std::array<uint8_t, 240>;

// Deletes its temporary file when the test ends.
class TempFile {
public:
    explicit TempFile(const std::string& name)
        : path_(::testing::TempDir() + "basalt_trace_headers_test_" + name + ".segy") {}
    ~TempFile() { std::remove(path_.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

// Writes big-endian values at 1-based SEG-Y byte positions (the standard byte numbers).
void put16(RawHeader& h, size_t first, uint16_t v) {
    h[first - 1] = static_cast<uint8_t>(v >> 8);
    h[first] = static_cast<uint8_t>(v & 0xFFu);
}

void put32(RawHeader& h, size_t first, uint32_t v) {
    h[first - 1] = static_cast<uint8_t>(v >> 24);
    h[first] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    h[first + 1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    h[first + 2] = static_cast<uint8_t>(v & 0xFFu);
}

void put_i16(RawHeader& h, size_t first, int16_t v) { put16(h, first, static_cast<uint16_t>(v)); }
void put_i32(RawHeader& h, size_t first, int32_t v) { put32(h, first, static_cast<uint32_t>(v)); }

// Header filled with 0xAB, so any read of a byte that holds no field would change a result.
RawHeader filled_header() {
    RawHeader h;
    h.fill(0xAB);
    put16(h, 115, 0);  // trace sample count: 0 leaves the count to the binary header
    return h;
}

// Sets every field the decoder reads, so a header with known values can be checked against them.
struct Fields {
    int32_t trace_seq = 0;
    int32_t field_record = 0;
    int32_t channel = 0;
    int32_t offset = 0;
    int32_t recv_elev = 0;
    int16_t scalar_elev = 0;
    int16_t scalar_coord = 0;
    int32_t source_x = 0;
    int32_t source_y = 0;
    int32_t group_x = 0;
    int32_t group_y = 0;
    int16_t coord_units = 0;
    int16_t delay = 0;
    int16_t year = 0;
    int16_t day = 0;
    int16_t hour = 0;
    int16_t minute = 0;
    int16_t second = 0;
};

RawHeader make_header(const Fields& f) {
    RawHeader h = filled_header();
    put_i32(h, 5, f.trace_seq);
    put_i32(h, 9, f.field_record);
    put_i32(h, 13, f.channel);
    put_i32(h, 37, f.offset);
    put_i32(h, 41, f.recv_elev);
    put_i16(h, 69, f.scalar_elev);
    put_i16(h, 71, f.scalar_coord);
    put_i32(h, 73, f.source_x);
    put_i32(h, 77, f.source_y);
    put_i32(h, 81, f.group_x);
    put_i32(h, 85, f.group_y);
    put_i16(h, 89, f.coord_units);
    put_i16(h, 109, f.delay);
    put_i16(h, 157, f.year);
    put_i16(h, 159, f.day);
    put_i16(h, 161, f.hour);
    put_i16(h, 163, f.minute);
    put_i16(h, 165, f.second);
    return h;
}

// Writes a SEG-Y file with fixed-length traces. Each trace is its header followed by `samples`,
// which must hold nsamples samples of the given format.
void write_file(const std::string& path, uint16_t format, uint16_t nsamples,
                const std::vector<RawHeader>& headers, const std::vector<uint8_t>& samples) {
    std::vector<uint8_t> bin(400, 0);
    bin[16] = 0x0F;  // bytes 3217-3218: sample interval 4000 us
    bin[17] = 0xA0;
    bin[20] = static_cast<uint8_t>(nsamples >> 8);  // bytes 3221-3222
    bin[21] = static_cast<uint8_t>(nsamples & 0xFFu);
    bin[24] = static_cast<uint8_t>(format >> 8);    // bytes 3225-3226
    bin[25] = static_cast<uint8_t>(format & 0xFFu);
    bin[300] = 0x01;                                // bytes 3501-3502: revision 1.0
    bin[302] = 0x00;                                // bytes 3503-3504: fixed length traces
    bin[303] = 0x01;

    std::vector<uint8_t> out(3200, ' ');
    out.insert(out.end(), bin.begin(), bin.end());
    for (const auto& h : headers) {
        out.insert(out.end(), h.begin(), h.end());
        out.insert(out.end(), samples.begin(), samples.end());
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    ASSERT_TRUE(file.good()) << "could not write " << path;
}

std::vector<uint8_t> be32_words(const std::vector<uint32_t>& words) {
    std::vector<uint8_t> out;
    for (uint32_t w : words) {
        out.push_back(static_cast<uint8_t>(w >> 24));
        out.push_back(static_cast<uint8_t>((w >> 16) & 0xFFu));
        out.push_back(static_cast<uint8_t>((w >> 8) & 0xFFu));
        out.push_back(static_cast<uint8_t>(w & 0xFFu));
    }
    return out;
}

// Expects read_trace_headers to throw std::runtime_error whose message contains `needle`.
void expect_headers_error(const std::string& path, const std::string& needle) {
    try {
        (void)read_trace_headers(path);
        ADD_FAILURE() << "read_trace_headers did not throw for " << path;
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find(needle), std::string::npos) << "message was: " << msg;
    }
}

} // namespace

// --- Field decoding ---

TEST(TraceHeadersTest, DecodesEachFieldFromItsBytes) {
    Fields f;
    f.trace_seq = 0x01020304;
    f.field_record = -7;
    f.channel = 42;
    f.offset = -123456;
    f.recv_elev = 4321;
    f.scalar_elev = 0;
    f.scalar_coord = 0;
    f.source_x = 11;
    f.source_y = -22;
    f.group_x = 33;
    f.group_y = -44;
    f.coord_units = 3;
    f.delay = -250;
    f.year = 2026;
    f.day = 286;
    f.hour = 23;
    f.minute = 59;
    f.second = 58;

    const TraceHeaders h = decode_trace_headers({make_header(f)});
    ASSERT_EQ(h.trace_sequence.size(), 1u);
    EXPECT_EQ(h.trace_sequence[0], 0x01020304);
    EXPECT_EQ(h.field_record[0], -7);
    EXPECT_EQ(h.channel[0], 42);
    EXPECT_EQ(h.offset[0], -123456);
    EXPECT_EQ(h.scalar_elev[0], 0);
    EXPECT_EQ(h.scalar_coord[0], 0);
    EXPECT_EQ(h.coord_units[0], 3);
    EXPECT_DOUBLE_EQ(h.source_x[0], 11.0);
    EXPECT_DOUBLE_EQ(h.source_y[0], -22.0);
    EXPECT_DOUBLE_EQ(h.group_x[0], 33.0);
    EXPECT_DOUBLE_EQ(h.group_y[0], -44.0);
    EXPECT_DOUBLE_EQ(h.receiver_elevation[0], 4321.0);
    EXPECT_EQ(h.delay_ms[0], -250);
    EXPECT_EQ(h.year[0], 2026);
    EXPECT_EQ(h.day_of_year[0], 286);
    EXPECT_EQ(h.hour[0], 23);
    EXPECT_EQ(h.minute[0], 59);
    EXPECT_EQ(h.second[0], 58);
}

TEST(TraceHeadersTest, Int32AndInt16ExtremesDecodeExactly) {
    Fields f;
    f.trace_seq = std::numeric_limits<int32_t>::max();
    f.offset = std::numeric_limits<int32_t>::min();
    f.field_record = std::numeric_limits<int32_t>::min();
    f.delay = std::numeric_limits<int16_t>::min();
    f.year = std::numeric_limits<int16_t>::max();
    f.coord_units = std::numeric_limits<int16_t>::min();

    const TraceHeaders h = decode_trace_headers({make_header(f)});
    EXPECT_EQ(h.trace_sequence[0], std::numeric_limits<int32_t>::max());
    EXPECT_EQ(h.offset[0], std::numeric_limits<int32_t>::min());
    EXPECT_EQ(h.field_record[0], std::numeric_limits<int32_t>::min());
    EXPECT_EQ(h.delay_ms[0], std::numeric_limits<int16_t>::min());
    EXPECT_EQ(h.year[0], std::numeric_limits<int16_t>::max());
    EXPECT_EQ(h.coord_units[0], std::numeric_limits<int16_t>::min());
}

// --- Scalars ---

TEST(TraceHeadersTest, PositiveScalarMultipliesCoordinatesAndElevation) {
    Fields f;
    f.scalar_coord = 10;
    f.source_x = -12;
    f.group_y = 7;
    f.scalar_elev = 2;
    f.recv_elev = 500;

    const TraceHeaders h = decode_trace_headers({make_header(f)});
    EXPECT_DOUBLE_EQ(h.source_x[0], -120.0);
    EXPECT_DOUBLE_EQ(h.group_y[0], 70.0);
    EXPECT_DOUBLE_EQ(h.receiver_elevation[0], 1000.0);
    EXPECT_EQ(h.scalar_coord[0], 10);
    EXPECT_EQ(h.scalar_elev[0], 2);
}

TEST(TraceHeadersTest, NegativeScalarDividesByItsAbsoluteValue) {
    Fields f;
    f.scalar_coord = -100;
    f.group_x = 12345;
    f.source_y = -1;
    f.scalar_elev = -10;
    f.recv_elev = -15;

    const TraceHeaders h = decode_trace_headers({make_header(f)});
    EXPECT_DOUBLE_EQ(h.group_x[0], 123.45);
    EXPECT_DOUBLE_EQ(h.source_y[0], -0.01);
    EXPECT_DOUBLE_EQ(h.receiver_elevation[0], -1.5);
    EXPECT_EQ(h.scalar_coord[0], -100);
    EXPECT_EQ(h.scalar_elev[0], -10);
}

TEST(TraceHeadersTest, ZeroScalarMeansOne) {
    Fields f;
    f.scalar_coord = 0;
    f.group_x = 987654;
    f.source_x = -5;
    f.scalar_elev = 0;
    f.recv_elev = 42;

    const TraceHeaders h = decode_trace_headers({make_header(f)});
    EXPECT_DOUBLE_EQ(h.group_x[0], 987654.0);
    EXPECT_DOUBLE_EQ(h.source_x[0], -5.0);
    EXPECT_DOUBLE_EQ(h.receiver_elevation[0], 42.0);
}

TEST(TraceHeadersTest, ScalarBoundsApplyTheirExactMagnitude) {
    Fields f;
    f.scalar_coord = -32768;  // most negative int16: divide by 32768, not by a wrapped value
    f.source_x = 32767;
    f.scalar_elev = 32767;
    f.recv_elev = -2147483647 - 1;
    f.group_x = -2147483647 - 1;

    const TraceHeaders h = decode_trace_headers({make_header(f)});
    EXPECT_DOUBLE_EQ(h.source_x[0], 32767.0 / 32768.0);
    EXPECT_DOUBLE_EQ(h.group_x[0], -2147483648.0 / 32768.0);
    EXPECT_DOUBLE_EQ(h.receiver_elevation[0], -2147483648.0 * 32767.0);
}

TEST(TraceHeadersTest, CoordinateAndElevationScalarsAreIndependent) {
    Fields f;
    f.scalar_coord = -10;
    f.scalar_elev = 0;
    f.recv_elev = 1234;
    f.source_x = 5000;
    f.group_y = 5000;

    const TraceHeaders h = decode_trace_headers({make_header(f)});
    EXPECT_DOUBLE_EQ(h.receiver_elevation[0], 1234.0);  // scalar_elev is 0, so not divided by 10
    EXPECT_DOUBLE_EQ(h.source_x[0], 500.0);

    f.scalar_coord = 0;
    f.scalar_elev = 3;
    const TraceHeaders g = decode_trace_headers({make_header(f)});
    EXPECT_DOUBLE_EQ(g.source_x[0], 5000.0);            // scalar_coord is 0, not multiplied by 3
    EXPECT_DOUBLE_EQ(g.receiver_elevation[0], 3702.0);
}

TEST(TraceHeadersTest, RawScalarsAndCoordUnitsAreReportedAsStored) {
    std::vector<RawHeader> headers;
    const int16_t scalars[] = {-100, 0, 10};
    const int16_t units[] = {1, 2, 3, 4, 0, 5};
    for (size_t i = 0; i < 6; ++i) {
        Fields f;
        f.scalar_coord = scalars[i % 3];
        f.scalar_elev = scalars[(i + 1) % 3];
        f.coord_units = units[i];
        headers.push_back(make_header(f));
    }

    const TraceHeaders h = decode_trace_headers(headers);
    ASSERT_EQ(h.coord_units.size(), 6u);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(h.coord_units[i], units[i]) << "trace " << i;
        EXPECT_EQ(h.scalar_coord[i], scalars[i % 3]) << "trace " << i;
        EXPECT_EQ(h.scalar_elev[i], scalars[(i + 1) % 3]) << "trace " << i;
    }
}

// --- Containers ---

TEST(TraceHeadersTest, DecodesEveryTraceInOrder) {
    std::vector<RawHeader> headers;
    for (int32_t t = 0; t < 4; ++t) {
        Fields f;
        f.trace_seq = 100 + t;
        f.offset = 50 * t;
        f.scalar_coord = -10;
        f.group_x = 1000 * (t + 1);
        headers.push_back(make_header(f));
    }

    const TraceHeaders h = decode_trace_headers(headers);
    ASSERT_EQ(h.trace_sequence.size(), 4u);
    for (size_t t = 0; t < 4; ++t) {
        EXPECT_EQ(h.trace_sequence[t], 100 + static_cast<int32_t>(t));
        EXPECT_EQ(h.offset[t], 50 * static_cast<int32_t>(t));
        EXPECT_DOUBLE_EQ(h.group_x[t], 100.0 * static_cast<double>(t + 1));
    }
}

TEST(TraceHeadersTest, NoTracesGivesEmptyArrays) {
    const TraceHeaders h = decode_trace_headers({});
    EXPECT_TRUE(h.trace_sequence.empty());
    EXPECT_TRUE(h.source_x.empty());
    EXPECT_TRUE(h.receiver_elevation.empty());
    EXPECT_TRUE(h.second.empty());
}

// --- Reading files ---

TEST(TraceHeadersTest, ReadTraceHeadersReturnsTheHeaderOfEveryTrace) {
    std::vector<RawHeader> headers;
    for (int32_t t = 0; t < 3; ++t) {
        Fields f;
        f.trace_seq = t + 1;
        f.field_record = 9001;
        f.channel = t + 1;
        f.offset = 25 * t;
        f.scalar_coord = -100;
        f.group_x = 1500 + t;
        f.coord_units = 2;
        headers.push_back(make_header(f));
    }
    TempFile file("headers_of_each_trace");
    write_file(file.path(), 5, 4, headers, std::vector<uint8_t>(16, 0));

    const TraceHeaders h = read_trace_headers(file.path());
    ASSERT_EQ(h.trace_sequence.size(), 3u);
    for (size_t t = 0; t < 3; ++t) {
        EXPECT_EQ(h.trace_sequence[t], static_cast<int32_t>(t) + 1);
        EXPECT_EQ(h.field_record[t], 9001);
        EXPECT_EQ(h.channel[t], static_cast<int32_t>(t) + 1);
        EXPECT_EQ(h.offset[t], 25 * static_cast<int32_t>(t));
        EXPECT_EQ(h.coord_units[t], 2);
        EXPECT_DOUBLE_EQ(h.group_x[t], (1500.0 + static_cast<double>(t)) / 100.0);
    }
}

TEST(TraceHeadersTest, ReadTraceHeadersDoesNotDecodeSamples) {
    // IBM word 0x7F123456 is far beyond the IEEE range. read_segy refuses such a file, but its
    // headers are valid and read_trace_headers never decodes the samples, so it must succeed.
    Fields f;
    f.trace_seq = 77;
    f.group_y = -600;
    f.scalar_coord = -6;
    TempFile file("headers_only_ibm");
    write_file(file.path(), 1, 2, {make_header(f)}, be32_words({0x41100000u, 0x7F123456u}));

    try {
        (void)read_segy(file.path());
        ADD_FAILURE() << "read_segy should reject the out-of-range IBM sample";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("IEEE single-precision range"), std::string::npos);
    }

    const TraceHeaders h = read_trace_headers(file.path());
    ASSERT_EQ(h.trace_sequence.size(), 1u);
    EXPECT_EQ(h.trace_sequence[0], 77);
    EXPECT_DOUBLE_EQ(h.group_y[0], -100.0);
}

TEST(TraceHeadersTest, ReadSegyHeadersKeepsHeadersAndSkipsSamples) {
    Fields f;
    f.trace_seq = 5;
    f.scalar_coord = -10;
    f.source_x = 2500;
    TempFile file("segy_headers_only");
    write_file(file.path(), 5, 3, {make_header(f), make_header(f)},
               be32_words({0x3F800000u, 0x40000000u, 0x40400000u}));

    const SegyData data = read_segy_headers(file.path());
    EXPECT_EQ(data.ntraces, 2u);
    EXPECT_EQ(data.nsamples, 3u);
    EXPECT_EQ(data.format, SampleFormat::IEEE32);
    EXPECT_NEAR(data.dt, 0.004, 1e-12);
    EXPECT_TRUE(data.samples.empty());
    ASSERT_EQ(data.raw_trace_headers.size(), 2u);
    EXPECT_EQ(data.raw_trace_headers[0], make_header(f));
    EXPECT_EQ(data.raw_trace_headers[1], make_header(f));

    const TraceHeaders h = decode_trace_headers(data.raw_trace_headers);
    EXPECT_DOUBLE_EQ(h.source_x[1], 250.0);
}

TEST(TraceHeadersTest, ReadTraceHeadersOfFileWithNoTraces) {
    TempFile file("no_traces");
    write_file(file.path(), 5, 4, {}, {});

    const TraceHeaders h = read_trace_headers(file.path());
    EXPECT_TRUE(h.trace_sequence.empty());
    EXPECT_TRUE(h.group_x.empty());
}

TEST(TraceHeadersTest, ReadTraceHeadersStillValidatesTheLayout) {
    Fields f;
    f.trace_seq = 1;

    // A header that declares a different sample count from the binary header is malformed.
    RawHeader bad = make_header(f);
    put16(bad, 115, 3);
    TempFile mismatch("headers_sample_mismatch");
    write_file(mismatch.path(), 5, 2, {bad}, be32_words({0u, 0u}));
    expect_headers_error(mismatch.path(), "declares 3 samples");

    // A file that ends inside a trace is malformed.
    TempFile truncated("headers_truncated");
    write_file(truncated.path(), 5, 2, {make_header(f)}, be32_words({0u, 0u}));
    {
        std::vector<char> bytes;
        {
            std::ifstream in(truncated.path(), std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        bytes.pop_back();
        std::ofstream out(truncated.path(), std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    expect_headers_error(truncated.path(), "does not match");
}

TEST(TraceHeadersTest, ReadTraceHeadersOfMissingFileThrows) {
    expect_headers_error(::testing::TempDir() + "basalt_no_such_trace_headers_file.segy", "cannot open");
}
