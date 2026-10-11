#include "basalt/io/segy_file.hpp"
#include "basalt/utils/ebcdic.hpp"
#include "basalt/utils/ibm_float.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::io;

namespace {

// Deletes its temporary file when the test ends.
class TempFile {
public:
    explicit TempFile(const std::string& name)
        : path_(::testing::TempDir() + "basalt_segy_file_test_" + name + ".segy") {}
    ~TempFile() { std::remove(path_.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

// --- Byte-level builders. These write big-endian fields independently of the implementation. ---

void set_be16(std::vector<uint8_t>& v, size_t off, uint16_t x) {
    v[off] = static_cast<uint8_t>(x >> 8);
    v[off + 1] = static_cast<uint8_t>(x & 0xFFu);
}

void set_be32(std::vector<uint8_t>& v, size_t off, uint32_t x) {
    v[off] = static_cast<uint8_t>(x >> 24);
    v[off + 1] = static_cast<uint8_t>((x >> 16) & 0xFFu);
    v[off + 2] = static_cast<uint8_t>((x >> 8) & 0xFFu);
    v[off + 3] = static_cast<uint8_t>(x & 0xFFu);
}

uint32_t get_be32(const std::vector<uint8_t>& v, size_t off) {
    return (static_cast<uint32_t>(v[off]) << 24) | (static_cast<uint32_t>(v[off + 1]) << 16) |
           (static_cast<uint32_t>(v[off + 2]) << 8) | static_cast<uint32_t>(v[off + 3]);
}

void append_be16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x & 0xFFu));
}

void append_be32(std::vector<uint8_t>& v, uint32_t x) {
    append_be16(v, static_cast<uint16_t>(x >> 16));
    append_be16(v, static_cast<uint16_t>(x & 0xFFFFu));
}

uint32_t float_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// 3200-byte textual header: ASCII text padded with spaces.
std::vector<uint8_t> ascii_text(const std::string& s) {
    std::vector<uint8_t> t(3200, ' ');
    for (size_t i = 0; i < s.size() && i < t.size(); ++i) {
        t[i] = static_cast<uint8_t>(s[i]);
    }
    return t;
}

// 400-byte binary header with the fields the reader uses; all other bytes are zero.
std::vector<uint8_t> binary_header(uint16_t format, uint16_t dt_us, uint16_t nsamples,
                                   int16_t ext_count = 0) {
    std::vector<uint8_t> b(400, 0);
    set_be16(b, 16, dt_us);                                   // bytes 3217-3218
    set_be16(b, 20, nsamples);                                // bytes 3221-3222
    set_be16(b, 24, format);                                  // bytes 3225-3226
    set_be16(b, 300, 0x0100);                                 // bytes 3501-3502 (rev 1.0)
    set_be16(b, 302, 1);                                      // bytes 3503-3504
    set_be16(b, 304, static_cast<uint16_t>(ext_count));       // bytes 3505-3506
    return b;
}

// 240-byte trace header followed by the encoded samples of that trace.
std::vector<uint8_t> trace(uint32_t seq, uint16_t samples_field, const std::vector<uint8_t>& sample_bytes) {
    std::vector<uint8_t> t(240, 0);
    set_be32(t, 0, seq);
    set_be32(t, 4, seq);
    set_be16(t, 114, samples_field);                          // bytes 115-116
    set_be16(t, 116, 4000);                                   // bytes 117-118
    t.insert(t.end(), sample_bytes.begin(), sample_bytes.end());
    return t;
}

std::vector<uint8_t> ieee_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> out;
    for (float v : values) append_be32(out, float_bits(v));
    return out;
}

std::vector<uint8_t> raw32_bytes(const std::vector<uint32_t>& words) {
    std::vector<uint8_t> out;
    for (uint32_t w : words) append_be32(out, w);
    return out;
}

std::vector<uint8_t> int32_bytes(const std::vector<int32_t>& values) {
    std::vector<uint8_t> out;
    for (int32_t v : values) append_be32(out, static_cast<uint32_t>(v));
    return out;
}

std::vector<uint8_t> int16_bytes(const std::vector<int16_t>& values) {
    std::vector<uint8_t> out;
    for (int16_t v : values) append_be16(out, static_cast<uint16_t>(v));
    return out;
}

std::vector<uint8_t> int8_bytes(const std::vector<int8_t>& values) {
    std::vector<uint8_t> out;
    for (int8_t v : values) out.push_back(static_cast<uint8_t>(v));
    return out;
}

// Concatenates textual, binary, extended and trace blocks into one file image.
std::vector<uint8_t> file_image(const std::vector<uint8_t>& text, const std::vector<uint8_t>& bin,
                                const std::vector<uint8_t>& ext,
                                const std::vector<std::vector<uint8_t>>& traces) {
    std::vector<uint8_t> out;
    out.insert(out.end(), text.begin(), text.end());
    out.insert(out.end(), bin.begin(), bin.end());
    out.insert(out.end(), ext.begin(), ext.end());
    for (const auto& t : traces) out.insert(out.end(), t.begin(), t.end());
    return out;
}

void write_all_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(out.good()) << "could not write " << path;
}

std::vector<uint8_t> read_all_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Expects read_segy to throw std::runtime_error whose message contains `needle`.
void expect_read_error(const std::string& path, const std::string& needle) {
    try {
        (void)read_segy(path);
        ADD_FAILURE() << "read_segy did not throw for " << path;
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find(needle), std::string::npos) << "message was: " << msg;
    }
}

// Deterministic pseudo-random samples in [-scale, scale].
std::vector<float> random_samples(size_t n, float scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

} // namespace

// --- Round trips ---

TEST(SegyFileTest, RoundTripIeee32IsExact) {
    const size_t ntraces = 7;
    const size_t nsamples = 33;
    const auto samples = random_samples(ntraces * nsamples, 1000.0f, 1234);
    const SegyData data = make_segy(samples, ntraces, nsamples, 0.004, SampleFormat::IEEE32);

    TempFile file("ieee_roundtrip");
    write_segy(file.path(), data, SampleFormat::IEEE32);

    const auto bytes = read_all_bytes(file.path());
    EXPECT_EQ(bytes.size(), 3600 + ntraces * (240 + nsamples * 4));

    const SegyData back = read_segy(file.path());
    EXPECT_EQ(back.format, SampleFormat::IEEE32);
    EXPECT_EQ(back.ntraces, ntraces);
    EXPECT_EQ(back.nsamples, nsamples);
    EXPECT_NEAR(back.dt, 0.004, 1e-12);
    EXPECT_EQ(back.samples, samples);
}

TEST(SegyFileTest, RoundTripIbm32WithinIbmPrecision) {
    // IBM32 keeps a 24-bit fraction after base-16 normalisation, so the relative error of
    // a truncated encode is below 2^-20 (about 9.5e-7).
    const size_t ntraces = 5;
    const size_t nsamples = 40;
    auto samples = random_samples(ntraces * nsamples, 10000.0f, 99);
    samples[0] = 0.0f;
    samples[1] = 1.0f;
    samples[2] = -1.0f;
    samples[3] = 0.5f;
    samples[4] = -100.0f;
    samples[5] = 12345.678f;
    samples[6] = 1e-3f;
    const SegyData data = make_segy(samples, ntraces, nsamples, 0.002, SampleFormat::IBM32);

    TempFile file("ibm_roundtrip");
    write_segy(file.path(), data, SampleFormat::IBM32);

    const SegyData back = read_segy(file.path());
    EXPECT_EQ(back.format, SampleFormat::IBM32);
    ASSERT_EQ(back.samples.size(), samples.size());
    EXPECT_EQ(back.samples[0], 0.0f);
    for (size_t i = 0; i < samples.size(); ++i) {
        const float orig = samples[i];
        const float got = back.samples[i];
        if (orig == 0.0f) {
            EXPECT_EQ(got, 0.0f) << "sample " << i;
        } else {
            EXPECT_LE(std::fabs(got - orig), 1e-6f * std::fabs(orig)) << "sample " << i << " orig " << orig;
        }
    }
}

TEST(SegyFileTest, IbmSamplesAreEncodedAsIbmWords) {
    // Exact IBM words for 1.0, -2.5, 0.5 and -100.0 appear big-endian at the start of the data.
    const SegyData data = make_segy({1.0f, -2.5f, 0.5f, -100.0f}, 1, 4, 0.001, SampleFormat::IBM32);
    TempFile file("ibm_words");
    write_segy(file.path(), data, SampleFormat::IBM32);

    const auto bytes = read_all_bytes(file.path());
    ASSERT_EQ(bytes.size(), 3600u + 240u + 16u);
    const size_t off = 3600 + 240;
    const std::vector<uint32_t> expected = {0x41100000u, 0xC1280000u, 0x40800000u, 0xC2640000u};
    for (size_t i = 0; i < expected.size(); ++i) {
        const uint32_t word = (static_cast<uint32_t>(bytes[off + 4 * i]) << 24) |
                              (static_cast<uint32_t>(bytes[off + 4 * i + 1]) << 16) |
                              (static_cast<uint32_t>(bytes[off + 4 * i + 2]) << 8) |
                              static_cast<uint32_t>(bytes[off + 4 * i + 3]);
        EXPECT_EQ(word, expected[i]) << "sample " << i;
    }
}

TEST(SegyFileTest, EmptyFileWithNoTracesRoundTrips) {
    const SegyData data = make_segy({}, 0, 8, 0.002, SampleFormat::IEEE32);
    TempFile file("empty");
    write_segy(file.path(), data);
    const SegyData back = read_segy(file.path());
    EXPECT_EQ(back.ntraces, 0u);
    EXPECT_EQ(back.nsamples, 8u);
    EXPECT_TRUE(back.samples.empty());
}

// --- Reading hand-built files in each supported format ---

TEST(SegyFileTest, ReadsIeee32Format5WithNegatives) {
    const std::vector<float> t0 = {1.0f, -2.5f, 0.5f, -100.0f};
    const std::vector<float> t1 = {-0.25f, 3.75f, -1000.0f, 0.0f};
    TempFile file("fmt5");
    write_all_bytes(file.path(), file_image(ascii_text("C 1 IEEE"), binary_header(5, 2000, 4), {},
                                            {trace(1, 4, ieee_bytes(t0)), trace(2, 4, ieee_bytes(t1))}));

    const SegyData d = read_segy(file.path());
    EXPECT_EQ(d.format, SampleFormat::IEEE32);
    EXPECT_EQ(d.ntraces, 2u);
    EXPECT_EQ(d.nsamples, 4u);
    std::vector<float> expected = t0;
    expected.insert(expected.end(), t1.begin(), t1.end());
    EXPECT_EQ(d.samples, expected);
}

TEST(SegyFileTest, ReadsIbm32Format1WithKnownValues) {
    // 1.0, -2.5, 0.5, -100.0 and -1.0, 2.5, 100.0, 0.0 in IBM 360 floating point.
    const std::vector<uint32_t> t0 = {0x41100000u, 0xC1280000u, 0x40800000u, 0xC2640000u};
    const std::vector<uint32_t> t1 = {0xC1100000u, 0x41280000u, 0x42640000u, 0x00000000u};
    TempFile file("fmt1");
    write_all_bytes(file.path(), file_image(ascii_text("C 1 IBM"), binary_header(1, 4000, 4), {},
                                            {trace(1, 4, raw32_bytes(t0)), trace(2, 4, raw32_bytes(t1))}));

    const SegyData d = read_segy(file.path());
    EXPECT_EQ(d.format, SampleFormat::IBM32);
    const std::vector<float> expected = {1.0f, -2.5f, 0.5f, -100.0f, -1.0f, 2.5f, 100.0f, 0.0f};
    EXPECT_EQ(d.samples, expected);
}

TEST(SegyFileTest, ReadsInexactIbm32WordsFromHandEncodedWords) {
    // IBM32 keeps a 24-bit fraction in units of 16^-6 with a base-16 exponent (value = f/2^24 *
    // 16^(e-64)), so most decimal values are rounded. The expected results are the exact fractions
    // of the stored words, written as integers over a power of two.
    //   0.1    -> 0x4019999A, fraction 0x19999A = 1677722  (1677722 / 2^24)
    //   -0.1   -> 0xC019999A, same fraction, negative sign (-1677722 / 2^24)
    //   1/3    -> 0x40555555, fraction 0x555555 = 5592405  (5592405 / 2^24)
    //   1234.5 -> 0x434D2800, exact: fraction 0x4D2800 / 2^24 * 16^3
    const std::vector<uint32_t> t0 = {0x4019999Au, 0xC019999Au, 0x40555555u, 0x434D2800u};
    TempFile file("fmt1_inexact");
    write_all_bytes(file.path(), file_image(ascii_text("C 1 IBM"), binary_header(1, 4000, 4), {},
                                            {trace(1, 4, raw32_bytes(t0))}));

    const SegyData d = read_segy(file.path());
    ASSERT_EQ(d.samples.size(), 4u);
    EXPECT_EQ(d.samples[0], 1677722.0f / 16777216.0f);
    EXPECT_EQ(d.samples[1], -1677722.0f / 16777216.0f);
    EXPECT_EQ(d.samples[2], 5592405.0f / 16777216.0f);
    EXPECT_EQ(d.samples[3], 1234.5f);
    // The rounded words stay within IBM32 precision (relative error below 2^-20) of the decimal values.
    EXPECT_NEAR(d.samples[0], 0.1, 0.1 * std::ldexp(1.0, -20));
    EXPECT_NEAR(d.samples[2], 1.0 / 3.0, (1.0 / 3.0) * std::ldexp(1.0, -20));
}

TEST(SegyFileTest, ReadsInt32Format2WithNegatives) {
    const std::vector<int32_t> t0 = {1, -2, 300000, -7};
    // Values are kept within float's exact integer range (2^24): beyond it the decoded float
    // rounds, and a comparison against a rounded expectation cannot detect an off-by-one.
    const std::vector<int32_t> t1 = {-16777215, 0, 42, -1};
    TempFile file("fmt2");
    write_all_bytes(file.path(), file_image(ascii_text("C 1 INT32"), binary_header(2, 4000, 4), {},
                                            {trace(1, 4, int32_bytes(t0)), trace(2, 4, int32_bytes(t1))}));

    const SegyData d = read_segy(file.path());
    EXPECT_EQ(d.format, SampleFormat::INT32);
    const std::vector<float> expected = {1.0f, -2.0f, 300000.0f, -7.0f,
                                         -16777215.0f, 0.0f, 42.0f, -1.0f};
    EXPECT_EQ(d.samples, expected);
}

TEST(SegyFileTest, ReadsInt16Format3WithNegatives) {
    const std::vector<int16_t> t0 = {1, -2, 300, -7};
    const std::vector<int16_t> t1 = {-32768, 32767, 0, -1};
    TempFile file("fmt3");
    write_all_bytes(file.path(), file_image(ascii_text("C 1 INT16"), binary_header(3, 4000, 4), {},
                                            {trace(1, 4, int16_bytes(t0)), trace(2, 4, int16_bytes(t1))}));

    const SegyData d = read_segy(file.path());
    EXPECT_EQ(d.format, SampleFormat::INT16);
    const std::vector<float> expected = {1.0f, -2.0f, 300.0f, -7.0f, -32768.0f, 32767.0f, 0.0f, -1.0f};
    EXPECT_EQ(d.samples, expected);
}

TEST(SegyFileTest, ReadsInt8Format8WithNegatives) {
    const std::vector<int8_t> t0 = {1, -2, 100, -7};
    const std::vector<int8_t> t1 = {-128, 127, 0, -1};
    TempFile file("fmt8");
    write_all_bytes(file.path(), file_image(ascii_text("C 1 INT8"), binary_header(8, 4000, 4), {},
                                            {trace(1, 4, int8_bytes(t0)), trace(2, 4, int8_bytes(t1))}));

    const SegyData d = read_segy(file.path());
    EXPECT_EQ(d.format, SampleFormat::INT8);
    const std::vector<float> expected = {1.0f, -2.0f, 100.0f, -7.0f, -128.0f, 127.0f, 0.0f, -1.0f};
    EXPECT_EQ(d.samples, expected);
}

// --- Header handling ---

TEST(SegyFileTest, DtAndNsamplesAreParsed) {
    const std::vector<float> samples(3 * 5, 0.25f);
    TempFile file("dt_nsamples");
    // First trace leaves its header sample count at zero (unset), which must be accepted.
    const std::vector<float> s0(samples.begin(), samples.begin() + 5);
    const std::vector<float> s1(samples.begin() + 5, samples.begin() + 10);
    const std::vector<float> s2(samples.begin() + 10, samples.end());
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(5, 2000, 5), {},
                                            {trace(1, 0, ieee_bytes(s0)), trace(2, 5, ieee_bytes(s1)),
                                             trace(3, 5, ieee_bytes(s2))}));

    const SegyData d = read_segy(file.path());
    EXPECT_NEAR(d.dt, 0.002, 1e-12);
    EXPECT_EQ(d.nsamples, 5u);
    EXPECT_EQ(d.ntraces, 3u);
    EXPECT_EQ(d.samples, samples);
    ASSERT_EQ(d.raw_trace_headers.size(), 3u);
}

TEST(SegyFileTest, AsciiTextHeaderIsDetected) {
    const std::string label = "C 1 ASCII TEST HEADER";
    const std::vector<uint8_t> text = ascii_text(label);
    TempFile file("ascii_text");
    write_all_bytes(file.path(), file_image(text, binary_header(5, 4000, 2), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));

    const SegyData d = read_segy(file.path());
    ASSERT_EQ(d.text_header.size(), 3200u);
    EXPECT_EQ(d.text_header.substr(0, label.size()), label);
    EXPECT_EQ(d.text_header, std::string(reinterpret_cast<const char*>(d.raw_text_header.data()),
                                         d.raw_text_header.size()));
}

TEST(SegyFileTest, EbcdicTextHeaderIsConvertedToAscii) {
    // "C 1 EBCDIC" in code page 037, padded with EBCDIC spaces (0x40).
    std::vector<uint8_t> text(3200, 0x40);
    const std::vector<uint8_t> label = {0xC3, 0x40, 0xF1, 0x40, 0xC5, 0xC2, 0xC3, 0xC4, 0xC9, 0xC3};
    std::copy(label.begin(), label.end(), text.begin());
    TempFile file("ebcdic_text");
    write_all_bytes(file.path(), file_image(text, binary_header(5, 4000, 2), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));

    const SegyData d = read_segy(file.path());
    ASSERT_EQ(d.text_header.size(), 3200u);
    EXPECT_EQ(d.text_header.substr(0, 10), "C 1 EBCDIC");
    EXPECT_EQ(d.text_header.substr(10), std::string(3190, ' '));
}

TEST(SegyFileTest, BlankEbcdicTextHeaderIsNotMistakenForAscii) {
    // An all-space EBCDIC header is 0x40 bytes, which are '@' in ASCII. It must decode to spaces.
    const std::vector<uint8_t> text(3200, 0x40);
    TempFile file("blank_ebcdic");
    write_all_bytes(file.path(), file_image(text, binary_header(5, 4000, 2), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));

    const SegyData d = read_segy(file.path());
    EXPECT_EQ(d.text_header, std::string(3200, ' '));
}

TEST(SegyFileTest, RawHeadersArePreservedOnWriteExceptFormatCode) {
    // Distinctive bytes everywhere except the fields the reader interprets, so that any
    // accidental rewrite of a header byte shows up.
    const size_t nsamples = 4;
    std::vector<uint8_t> text(3200);
    for (size_t i = 0; i < text.size(); ++i) text[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);

    std::vector<uint8_t> bin = binary_header(5, 4000, static_cast<uint16_t>(nsamples));
    for (size_t i = 0; i < bin.size(); ++i) {
        const bool is_field = (i >= 16 && i <= 17) || (i >= 20 && i <= 21) || (i >= 24 && i <= 25) ||
                              (i >= 300 && i <= 305);
        if (!is_field) bin[i] = static_cast<uint8_t>((i * 91 + 7) & 0xFF);
    }

    std::vector<uint8_t> hdr = trace(9, static_cast<uint16_t>(nsamples), {});
    for (size_t i = 0; i < 240; ++i) {
        if (i != 114 && i != 115) hdr[i] = static_cast<uint8_t>((i * 53 + 3) & 0xFF);
    }
    std::vector<uint8_t> hdr2 = trace(10, static_cast<uint16_t>(nsamples), {});
    for (size_t i = 0; i < 240; ++i) {
        if (i != 114 && i != 115) hdr2[i] = static_cast<uint8_t>((i * 29 + 101) & 0xFF);
    }

    const std::vector<float> values = {1.0f, -2.5f, 0.5f, -100.0f};
    const std::vector<uint8_t> payload = ieee_bytes(values);
    std::vector<uint8_t> t0(hdr.begin(), hdr.begin() + 240);
    t0.insert(t0.end(), payload.begin(), payload.end());
    std::vector<uint8_t> t1(hdr2.begin(), hdr2.begin() + 240);
    t1.insert(t1.end(), payload.begin(), payload.end());

    TempFile in_file("preserve_in");
    TempFile out_file("preserve_out");
    write_all_bytes(in_file.path(), file_image(text, bin, {}, {t0, t1}));

    const SegyData d = read_segy(in_file.path());
    EXPECT_EQ(d.format, SampleFormat::IEEE32);
    write_segy(out_file.path(), d, SampleFormat::IBM32);

    const auto out = read_all_bytes(out_file.path());
    const auto in = read_all_bytes(in_file.path());
    // IBM32 and IEEE32 both use 4-byte samples, so the file size is unchanged.
    ASSERT_EQ(out.size(), in.size());

    // Text header: byte-for-byte identical.
    EXPECT_TRUE(std::equal(text.begin(), text.end(), out.begin()));

    // Binary header: identical except the format code (bytes 3225-3226 -> index 24-25).
    // Compared in two ranges around the format code, so the expected header is never patched in place.
    EXPECT_TRUE(std::equal(bin.begin(), bin.begin() + 24, out.begin() + 3200));
    EXPECT_TRUE(std::equal(bin.begin() + 26, bin.end(), out.begin() + 3200 + 26));
    EXPECT_EQ(out[3200 + 24], 0x00);
    EXPECT_EQ(out[3200 + 25], 0x01);  // IBM32

    // Trace headers: identical.
    const size_t ibm_trace = 240 + nsamples * 4;
    EXPECT_TRUE(std::equal(hdr.begin(), hdr.begin() + 240, out.begin() + 3600));
    EXPECT_TRUE(std::equal(hdr2.begin(), hdr2.begin() + 240, out.begin() + 3600 + ibm_trace));

    // The re-read file keeps the values (all exactly representable in IBM32).
    const SegyData back = read_segy(out_file.path());
    EXPECT_EQ(back.format, SampleFormat::IBM32);
    EXPECT_EQ(back.raw_binary_header[25], 0x01);
    EXPECT_EQ(back.samples.size(), 2 * nsamples);
    EXPECT_EQ(back.samples[0], 1.0f);
    EXPECT_EQ(back.samples[1], -2.5f);
    EXPECT_EQ(back.samples[7], -100.0f);
}

TEST(SegyFileTest, ExtendedTextualHeadersAreSkipped) {
    // Two extended headers of junk. If they were not skipped, the samples would be misread.
    std::vector<uint8_t> ext(2 * 3200, 'X');
    const std::vector<float> t0 = {0.5f, -0.5f, 2.0f};
    const std::vector<float> t1 = {-3.0f, 4.0f, -5.0f};
    TempFile file("extended");
    write_all_bytes(file.path(), file_image(ascii_text("C 1 MAIN"), binary_header(5, 4000, 3, 2), ext,
                                            {trace(1, 3, ieee_bytes(t0)), trace(2, 3, ieee_bytes(t1))}));

    const SegyData d = read_segy(file.path());
    EXPECT_EQ(d.ntraces, 2u);
    EXPECT_EQ(d.nsamples, 3u);
    EXPECT_EQ(d.text_header.substr(0, 6), "C 1 MA");
    std::vector<float> expected = t0;
    expected.insert(expected.end(), t1.begin(), t1.end());
    EXPECT_EQ(d.samples, expected);
}

// --- Error handling ---

TEST(SegyFileTest, TruncatedFileThrowsNamingFile) {
    TempFile file("truncated");
    auto bytes = file_image(ascii_text("C 1"), binary_header(5, 4000, 4), {},
                            {trace(1, 4, ieee_bytes({1.0f, 2.0f, 3.0f, 4.0f})),
                             trace(2, 4, ieee_bytes({5.0f, 6.0f, 7.0f, 8.0f}))});
    bytes.resize(bytes.size() - 10);
    write_all_bytes(file.path(), bytes);
    expect_read_error(file.path(), file.path());
}

TEST(SegyFileTest, WrongSizeFileThrowsNamingFile) {
    TempFile file("wrong_size");
    auto bytes = file_image(ascii_text("C 1"), binary_header(5, 4000, 4), {},
                            {trace(1, 4, ieee_bytes({1.0f, 2.0f, 3.0f, 4.0f}))});
    bytes.resize(bytes.size() + 7, 0);
    write_all_bytes(file.path(), bytes);
    expect_read_error(file.path(), file.path());
}

TEST(SegyFileTest, FileShorterThanHeadersThrows) {
    TempFile file("short");
    write_all_bytes(file.path(), std::vector<uint8_t>(100, 0));
    expect_read_error(file.path(), file.path());
}

TEST(SegyFileTest, NonexistentFileThrows) {
    const std::string path = ::testing::TempDir() + "basalt_segy_file_test_does_not_exist.segy";
    std::remove(path.c_str());
    EXPECT_THROW((void)read_segy(path), std::runtime_error);
}

TEST(SegyFileTest, VariableExtendedHeaderCountIsUnsupported) {
    TempFile file("ext_minus_one");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(5, 4000, 2, -1), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));
    expect_read_error(file.path(), "unsupported");
}

TEST(SegyFileTest, UnsupportedFormatCodeThrows) {
    // Format 4 (fixed point with gain) is not in the supported set.
    TempFile file("fmt4");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(4, 4000, 2), {},
                                            {trace(1, 2, raw32_bytes({0u, 0u}))}));
    expect_read_error(file.path(), "unsupported");
}

TEST(SegyFileTest, TraceHeaderSampleCountMismatchThrows) {
    TempFile file("trace_mismatch");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(5, 4000, 5), {},
                                            {trace(1, 5, ieee_bytes({1, 2, 3, 4, 5})),
                                             trace(2, 7, ieee_bytes({1, 2, 3, 4, 5}))}));
    expect_read_error(file.path(), "trace 1");
}

TEST(SegyFileTest, ZeroSamplesPerTraceThrows) {
    TempFile file("zero_samples");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(5, 4000, 0), {}, {}));
    expect_read_error(file.path(), file.path());
}

TEST(SegyFileTest, WriteRejectsUnsupportedOutputFormat) {
    const SegyData data = make_segy({1.0f, 2.0f}, 1, 2, 0.001, SampleFormat::IEEE32);
    TempFile file("bad_out_format");
    EXPECT_THROW(write_segy(file.path(), data, SampleFormat::INT16), std::invalid_argument);
}

TEST(SegyFileTest, WriteRejectsSampleCountMismatch) {
    // The samples vector is sized for nsamples = 4, so only the binary-header check can fire.
    SegyData data = make_segy({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, 2, 4, 0.001,
                              SampleFormat::IEEE32);
    data.raw_binary_header[20] = 0;  // bytes 3221-3222 hold 2 samples per trace
    data.raw_binary_header[21] = 2;
    TempFile file("nsamples_mismatch");
    EXPECT_THROW(write_segy(file.path(), data, SampleFormat::IEEE32), std::invalid_argument);
}

TEST(SegyFileTest, WriteRejectsTraceSampleCountMismatch) {
    SegyData data = make_segy({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, 2, 4, 0.001,
                              SampleFormat::IEEE32);
    data.raw_trace_headers[1][114] = 0;
    data.raw_trace_headers[1][115] = 3;  // trace 1 header says 3 samples
    TempFile file("trace_nsamples_mismatch");
    EXPECT_THROW(write_segy(file.path(), data, SampleFormat::IEEE32), std::invalid_argument);
}

TEST(SegyFileTest, MakeSegyRejectsBadArguments) {
    EXPECT_THROW(make_segy({1.0f, 2.0f, 3.0f}, 2, 2, 0.001), std::invalid_argument);
    EXPECT_THROW(make_segy({1.0f, 2.0f}, 1, 2, 0.0), std::invalid_argument);
    EXPECT_THROW(make_segy({1.0f, 2.0f}, 1, 2, 0.001, SampleFormat::INT16), std::invalid_argument);
    EXPECT_THROW(make_segy({}, 1, 0, 0.001), std::invalid_argument);
}

// --- make_segy header layout, checked in the written file ---

TEST(SegyFileTest, MakeSegyWritesValidHeaders) {
    const size_t ntraces = 2;
    const size_t nsamples = 6;
    const SegyData data = make_segy(std::vector<float>(ntraces * nsamples, 0.5f), ntraces, nsamples, 0.002,
                                    SampleFormat::IEEE32);
    TempFile file("make_headers");
    write_segy(file.path(), data, SampleFormat::IEEE32);
    const auto b = read_all_bytes(file.path());
    ASSERT_EQ(b.size(), 3600 + ntraces * (240 + nsamples * 4));

    // Text header: EBCDIC "C 1 CREATED BY BASALT", then EBCDIC spaces.
    EXPECT_EQ(b[0], 0xC3);  // C
    EXPECT_EQ(b[1], 0x40);  // space
    EXPECT_EQ(b[2], 0xF1);  // 1
    EXPECT_EQ(b[3], 0x40);
    EXPECT_EQ(b[3199], 0x40);
    // "C 1 CREATED BY BASALT" in EBCDIC code page 037.
    const std::vector<uint8_t> label = {0xC3, 0x40, 0xF1, 0x40, 0xC3, 0xD9, 0xC5, 0xC1, 0xE3, 0xC5, 0xC4,
                                        0x40, 0xC2, 0xE8, 0x40, 0xC2, 0xC1, 0xE2, 0xC1, 0xD3, 0xE3};
    EXPECT_TRUE(std::equal(label.begin(), label.end(), b.begin()));

    // Binary header (file offset = 3200 + binary index).
    EXPECT_EQ(b[3200 + 16], 0x07);  // sample interval 2000 us = 0x07D0
    EXPECT_EQ(b[3200 + 17], 0xD0);
    EXPECT_EQ(b[3200 + 20], 0x00);  // samples per trace = 6
    EXPECT_EQ(b[3200 + 21], 0x06);
    EXPECT_EQ(b[3200 + 24], 0x00);  // format 5 = IEEE32
    EXPECT_EQ(b[3200 + 25], 0x05);
    EXPECT_EQ(b[3200 + 300], 0x01);  // revision 0x0100 at bytes 3501-3502
    EXPECT_EQ(b[3200 + 301], 0x00);
    EXPECT_EQ(b[3200 + 302], 0x00);  // fixed-length flag 1 at bytes 3503-3504
    EXPECT_EQ(b[3200 + 303], 0x01);

    // Trace headers: sequence numbers, samples and dt.
    const size_t t1 = 3600 + 240 + nsamples * 4;
    EXPECT_EQ(get_be32(b, 3600 + 0), 1u);
    EXPECT_EQ(get_be32(b, 3600 + 4), 1u);
    EXPECT_EQ(get_be32(b, t1 + 0), 2u);
    EXPECT_EQ(get_be32(b, t1 + 4), 2u);
    EXPECT_EQ((static_cast<uint32_t>(b[3600 + 114]) << 8) | b[3600 + 115], nsamples);
    EXPECT_EQ((static_cast<uint32_t>(b[3600 + 116]) << 8) | b[3600 + 117], 2000u);
    EXPECT_EQ((static_cast<uint32_t>(b[t1 + 114]) << 8) | b[t1 + 115], nsamples);
}

// --- Edge cases ---

TEST(SegyFileTest, NulPaddedAsciiTextHeaderIsDetected) {
    // NUL bytes are neutral, so an ASCII label followed by NUL padding is still read as ASCII.
    const std::string label = "C 1 NUL PADDED ASCII HEADER";
    std::vector<uint8_t> text = ascii_text(label);
    std::fill(text.begin() + static_cast<std::ptrdiff_t>(label.size()), text.end(), uint8_t{0});
    TempFile file("nul_padded_text");
    write_all_bytes(file.path(), file_image(text, binary_header(5, 4000, 2), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));

    const SegyData d = read_segy(file.path());
    ASSERT_EQ(d.text_header.size(), 3200u);
    EXPECT_EQ(d.text_header.substr(0, label.size()), label);
    EXPECT_EQ(d.samples, (std::vector<float>{1.0f, 2.0f}));
}

TEST(SegyFileTest, ZeroSampleIntervalThrows) {
    TempFile file("zero_interval");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(5, 0, 2), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));
    expect_read_error(file.path(), "sample interval");
}

TEST(SegyFileTest, ExtendedHeaderCountIsZeroedOnWrite) {
    // The input declares two extended headers. They are skipped on read and not retained, so the
    // written file must not claim them: its count is zero and its layout has no extended headers.
    const size_t nsamples = 3;
    const std::vector<float> t0 = {0.5f, -0.5f, 2.0f};
    const std::vector<float> t1 = {-3.0f, 4.0f, -5.0f};
    TempFile in_file("ext_in");
    TempFile out_file("ext_out");
    write_all_bytes(in_file.path(),
                    file_image(ascii_text("C 1 MAIN"), binary_header(5, 4000, 3, 2),
                               std::vector<uint8_t>(2 * 3200, 'X'),
                               {trace(1, 3, ieee_bytes(t0)), trace(2, 3, ieee_bytes(t1))}));

    const SegyData d = read_segy(in_file.path());
    write_segy(out_file.path(), d, SampleFormat::IEEE32);

    const auto out = read_all_bytes(out_file.path());
    ASSERT_EQ(out.size(), 3600 + 2 * (240 + nsamples * 4));
    EXPECT_EQ((static_cast<uint32_t>(out[3200 + 304]) << 8) | out[3200 + 305], 0u);
    const SegyData back = read_segy(out_file.path());
    EXPECT_EQ(back.ntraces, 2u);
    EXPECT_EQ(back.samples, d.samples);
}

TEST(SegyFileTest, IbmWriteRejectsNonFiniteSamples) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float bad : {inf, -inf, nan}) {
        const SegyData data = make_segy({1.0f, bad}, 1, 2, 0.001, SampleFormat::IEEE32);
        TempFile file("ibm_nonfinite");
        std::remove(file.path().c_str());
        EXPECT_THROW(write_segy(file.path(), data, SampleFormat::IBM32), std::invalid_argument);
        // The check runs before the output file is created.
        std::ifstream probe(file.path());
        EXPECT_FALSE(probe.good()) << "output file should not exist for value " << bad;
    }
}

TEST(SegyFileTest, IeeeWriteKeepsNonFiniteAndSubnormalValues) {
    const float inf = std::numeric_limits<float>::infinity();
    const float subnormal = 1.4e-45f;
    const SegyData data = make_segy({inf, -inf, subnormal, -subnormal}, 1, 4, 0.001, SampleFormat::IEEE32);
    TempFile file("ieee_special");
    write_segy(file.path(), data, SampleFormat::IEEE32);
    const SegyData back = read_segy(file.path());
    EXPECT_EQ(back.samples, data.samples);
}

TEST(SegyFileTest, IbmWriteFlushesSubnormalsToZero) {
    // IEEE subnormals (below 1.2e-38) are written as zero words. ieee_to_ibm would otherwise write
    // a nonzero word that is mis-scaled. The raw words are checked because the reader's own
    // underflow handling would hide the difference after decoding.
    const SegyData data = make_segy({1.4e-45f, -1.4e-45f, 1.0e-40f, 1.0f}, 1, 4, 0.001, SampleFormat::IEEE32);
    TempFile file("ibm_subnormal");
    write_segy(file.path(), data, SampleFormat::IBM32);
    const auto bytes = read_all_bytes(file.path());
    ASSERT_EQ(bytes.size(), 3600u + 240u + 16u);
    const size_t off = 3600 + 240;
    EXPECT_EQ(get_be32(bytes, off + 0), 0u);
    EXPECT_EQ(get_be32(bytes, off + 4), 0u);
    EXPECT_EQ(get_be32(bytes, off + 8), 0u);
    EXPECT_EQ(get_be32(bytes, off + 12), 0x41100000u);  // 1.0 in IBM32
}

TEST(SegyFileTest, NegativeExtendedHeaderCountOtherThanMinusOneIsRejected) {
    TempFile file("ext_minus_two");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(5, 4000, 2, -2), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));
    expect_read_error(file.path(), "invalid extended textual header count");
}

TEST(SegyFileTest, ExtendedHeaderBlockPastEndOfFileIsTruncated) {
    // Declares two extended headers, but the file holds only one before its traces.
    TempFile file("ext_truncated");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(5, 4000, 2, 2),
                                            std::vector<uint8_t>(3200, 'X'),
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));
    expect_read_error(file.path(), "file is truncated");
}

TEST(SegyFileTest, WriteRejectsMissingTraceHeader) {
    SegyData data = make_segy({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, 2, 4, 0.001,
                              SampleFormat::IEEE32);
    data.raw_trace_headers.pop_back();  // one header for two traces
    TempFile file("missing_trace_header");
    EXPECT_THROW(write_segy(file.path(), data, SampleFormat::IEEE32), std::invalid_argument);
}

TEST(SegyFileTest, IbmWordBeyondIeeeRangeIsRejectedOnRead) {
    // 0x61800000 is 2^131 and 0x7F123456 is near 16^190. Neither fits in IEEE single precision,
    // so the reader must refuse the file instead of returning infinity.
    TempFile file("ibm_overflow");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(1, 4000, 2), {},
                                            {trace(1, 2, raw32_bytes({0x41100000u, 0x61800000u}))}));
    expect_read_error(file.path(), "IEEE single-precision range");

    TempFile file2("ibm_overflow_large");
    write_all_bytes(file2.path(), file_image(ascii_text("C 1"), binary_header(1, 4000, 1), {},
                                             {trace(1, 1, raw32_bytes({0x7F123456u}))}));
    expect_read_error(file2.path(), "IEEE single-precision range");
}

TEST(SegyFileTest, IbmLargestFiniteWordReadsAsFiniteValue) {
    // 0x60800000 is 0.5 * 16^32 = 2^127, the largest IBM word with a normalised fraction that fits
    // in IEEE single precision. It must read back exactly, not be rejected.
    TempFile file("ibm_range_edge");
    write_all_bytes(file.path(), file_image(ascii_text("C 1"), binary_header(1, 4000, 1), {},
                                            {trace(1, 1, raw32_bytes({0x60800000u}))}));
    const SegyData d = read_segy(file.path());
    ASSERT_EQ(d.samples.size(), 1u);
    EXPECT_TRUE(std::isfinite(d.samples[0]));
    EXPECT_EQ(d.samples[0], std::ldexp(1.0f, 127));
}

TEST(SegyFileTest, NonAsciiBytesInAsciiTextHeaderBecomeSpacesInTextHeader) {
    // A mostly-ASCII header with a few high bytes. text_header must be pure ASCII, while the raw
    // bytes are kept unchanged for write-back.
    std::vector<uint8_t> text = ascii_text("C 1 HIGH BYTE TEST");
    text[20] = 0xE9;
    text[30] = 0x80;
    TempFile file("ascii_high_bytes");
    write_all_bytes(file.path(), file_image(text, binary_header(5, 4000, 2), {},
                                            {trace(1, 2, ieee_bytes({1.0f, 2.0f}))}));

    const SegyData d = read_segy(file.path());
    ASSERT_EQ(d.text_header.size(), 3200u);
    EXPECT_EQ(d.text_header.substr(0, 18), "C 1 HIGH BYTE TEST");
    EXPECT_EQ(d.text_header[20], ' ');
    EXPECT_EQ(d.text_header[30], ' ');
    EXPECT_TRUE(std::all_of(d.text_header.begin(), d.text_header.end(),
                            [](char c) { return static_cast<unsigned char>(c) < 0x80; }));
    EXPECT_EQ(d.raw_text_header[20], 0xE9);
    EXPECT_EQ(d.raw_text_header[30], 0x80);
}

TEST(SegyFileTest, MakeSegyRejectsSampleCountWhoseProductWouldWrap) {
    // (2^63 + 1) * 2 wraps to 2 in size_t, which would match a two-element samples vector.
    // The size check must reject it with std::invalid_argument, not fail later in an allocation.
    EXPECT_THROW(make_segy({1.0f, 2.0f}, (size_t{1} << 63) + 1, 2, 0.001), std::invalid_argument);

    SegyData d = make_segy({1.0f, 2.0f}, 1, 2, 0.001, SampleFormat::IEEE32);
    d.ntraces = (size_t{1} << 63) + 1;
    TempFile file("wrap_write");
    EXPECT_THROW(write_segy(file.path(), d, SampleFormat::IEEE32), std::invalid_argument);
}

TEST(SegyFileTest, MakeSegyRejectsDtThatIsNotWholeMicroseconds) {
    // Sub-microsecond and fractional-microsecond intervals cannot be stored, so they are rejected
    // rather than silently rounded.
    EXPECT_THROW(make_segy({1.0f, 2.0f}, 1, 2, 0.0000015), std::invalid_argument);  // 1.5 us
    EXPECT_THROW(make_segy({1.0f, 2.0f}, 1, 2, 0.0001234), std::invalid_argument);  // 123.4 us
    EXPECT_THROW(make_segy({1.0f, 2.0f}, 1, 2, 1e-9), std::invalid_argument);       // below 1 us
    EXPECT_THROW(make_segy({1.0f, 2.0f}, 1, 2, 0.07), std::invalid_argument);       // above 65535 us
    EXPECT_NO_THROW(make_segy({1.0f, 2.0f}, 1, 2, 0.000001));
    EXPECT_NO_THROW(make_segy({1.0f, 2.0f}, 1, 2, 0.065535));
    EXPECT_NEAR(make_segy({1.0f, 2.0f}, 1, 2, 0.0005).dt, 0.0005, 1e-12);
}
