#include "basalt/utils/trace_utils.hpp"
#include "basalt/segy_format.hpp"
#include "basalt/utils/ebcdic.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <cstring>

using namespace basalt;
using namespace basalt::utils;

// --- Dead trace detection by header ---

TEST(TraceUtilsTest, DeadTraceByHeader) {
    SegyTraceHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    
    // trace_id_code == 2 means dead trace (SEG-Y Rev 1)
    hdr.trace_id_code = 2;
    EXPECT_TRUE(is_dead_trace(hdr));
}

TEST(TraceUtilsTest, LiveTraceByHeader) {
    SegyTraceHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    
    // trace_id_code == 1 means production (live) data
    hdr.trace_id_code = 1;
    EXPECT_FALSE(is_dead_trace(hdr));
}

TEST(TraceUtilsTest, OtherTraceCodesNotDead) {
    SegyTraceHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    
    // Various other codes should not be detected as dead
    int16_t codes[] = {0, 1, 3, 4, 5, 6, 7};
    for (int16_t code : codes) {
        hdr.trace_id_code = code;
        EXPECT_FALSE(is_dead_trace(hdr)) << "trace_id_code=" << code;
    }
}

// --- Zero trace detection ---

TEST(TraceUtilsTest, AllZeroSamples) {
    std::vector<float> samples(1000, 0.0f);
    EXPECT_TRUE(is_zero_trace(samples.data(), samples.size()));
}

TEST(TraceUtilsTest, NearZeroSamples) {
    std::vector<float> samples(1000, 1e-35f);
    EXPECT_TRUE(is_zero_trace(samples.data(), samples.size(), 1e-30f));
}

TEST(TraceUtilsTest, NonZeroTrace) {
    std::vector<float> samples(1000, 0.0f);
    // Place one non-zero sample in the middle
    samples[500] = 1.0f;
    EXPECT_FALSE(is_zero_trace(samples.data(), samples.size()));
}

TEST(TraceUtilsTest, EmptyTrace) {
    EXPECT_TRUE(is_zero_trace(nullptr, 0));
}

TEST(TraceUtilsTest, SingleSampleNonZero) {
    float sample = 42.0f;
    EXPECT_FALSE(is_zero_trace(&sample, 1));
}

TEST(TraceUtilsTest, AllNanTraceIsNotZero) {
    std::vector<float> samples(256, std::numeric_limits<float>::quiet_NaN());
    EXPECT_FALSE(is_zero_trace(samples.data(), samples.size()));
}

TEST(TraceUtilsTest, SingleNanAmongZerosIsNotZero) {
    std::vector<float> samples(256, 0.0f);
    samples[100] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(is_zero_trace(samples.data(), samples.size()));
}

TEST(TraceUtilsTest, InfinityIsNotZero) {
    std::vector<float> samples(16, 0.0f);
    samples[3] = -std::numeric_limits<float>::infinity();
    EXPECT_FALSE(is_zero_trace(samples.data(), samples.size()));
}

TEST(TraceUtilsTest, NegativeZeroSamplesAreZero) {
    std::vector<float> samples(16, -0.0f);
    EXPECT_TRUE(is_zero_trace(samples.data(), samples.size()));
}


// --- EBCDIC (cp037) decoding ---

namespace {
// EBCDIC cp037 encoding of every printable ASCII char 0x20..0x7E
// (generated with Python: ''.join(map(chr, range(0x20, 0x7F))).encode('cp037')).
constexpr uint8_t kCp037Printable[95] = {
    0x40, 0x5A, 0x7F, 0x7B, 0x5B, 0x6C, 0x50, 0x7D, 0x4D, 0x5D, 0x5C, 0x4E, 0x6B, 0x60, 0x4B, 0x61,
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0x7A, 0x5E, 0x4C, 0x7E, 0x6E, 0x6F,
    0x7C, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
    0xD7, 0xD8, 0xD9, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xBA, 0xE0, 0xBB, 0xB0, 0x6D,
    0x79, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xC0, 0x4F, 0xD0, 0xA1};
}

TEST(EbcdicTest, AllPrintableAsciiRoundTripsFromCp037) {
    std::string expected;
    for (int c = 0x20; c <= 0x7E; ++c) expected.push_back(static_cast<char>(c));
    EXPECT_EQ(EbcdicConverter::to_ascii(kCp037Printable, sizeof(kCp037Printable)), expected);
}

TEST(EbcdicTest, KnownCp037Mappings) {
    auto one = [](uint8_t b) { return EbcdicConverter::to_ascii(&b, 1)[0]; };
    EXPECT_EQ(one(0x40), ' ');
    EXPECT_EQ(one(0xC1), 'A');
    EXPECT_EQ(one(0x81), 'a');
    EXPECT_EQ(one(0xF0), '0');
    // These differ between cp037 and the table that used to be in the code
    EXPECT_EQ(one(0xB0), '^');   // was NUL
    EXPECT_EQ(one(0xBA), '[');   // was NUL
    EXPECT_EQ(one(0xBB), ']');   // was NUL
    EXPECT_EQ(one(0x4F), '|');
    EXPECT_EQ(one(0x5A), '!');
    EXPECT_EQ(one(0x5F), '?');   // NOT sign U+00AC, not '^'
    EXPECT_EQ(one(0x4A), '?');   // cent sign U+00A2, not '['
    EXPECT_EQ(one(0x6A), '?');   // broken bar U+00A6, not '|'
    EXPECT_EQ(one(0xAD), '?');   // Y-acute in cp037, not '['
    EXPECT_EQ(one(0xBD), '?');   // diaeresis in cp037, not ']'
    EXPECT_EQ(one(0xC0), '{');
    EXPECT_EQ(one(0xD0), '}');
    EXPECT_EQ(one(0xE0), '\\');
    EXPECT_EQ(one(0xA1), '~');
    EXPECT_EQ(one(0x79), '`');
    EXPECT_EQ(one(0x25), '\n');
    EXPECT_EQ(one(0x00), '\0');
}

TEST(EbcdicTest, NoNulForPrintableCharacters) {
    // Only the single input byte 0x00 may decode to NUL, in both encodings.
    for (int b = 0; b < 256; ++b) {
        const uint8_t byte = static_cast<uint8_t>(b);
        const std::string a = EbcdicConverter::to_ascii(&byte, 1);
        const std::string l = EbcdicConverter::to_latin1(&byte, 1);
        ASSERT_EQ(a.size(), 1u);
        ASSERT_EQ(l.size(), 1u);
        if (b == 0) {
            EXPECT_EQ(a[0], '\0');
            EXPECT_EQ(l[0], '\0');
        } else {
            EXPECT_NE(a[0], '\0') << "byte " << b;
            EXPECT_NE(l[0], '\0') << "byte " << b;
        }
        // ASCII output is always 7-bit
        EXPECT_GE(a[0], 0) << "byte " << b;
    }
}

TEST(EbcdicTest, Latin1KeepsAccentedLetters) {
    const uint8_t input[] = {0x51, 0x59, 0xC1, 0x5F};  // e-acute, sharp s, 'A', not-sign
    const std::string out = EbcdicConverter::to_latin1(input, sizeof(input));
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(static_cast<unsigned char>(out[0]), 0xE9u);
    EXPECT_EQ(static_cast<unsigned char>(out[1]), 0xDFu);
    EXPECT_EQ(out[2], 'A');
    EXPECT_EQ(static_cast<unsigned char>(out[3]), 0xACu);
    EXPECT_EQ(EbcdicConverter::to_ascii(input, sizeof(input)), "??A?");
}

TEST(EbcdicTest, TextHeaderLineKeepsLength) {
    // "C 1 CLIENT" followed by EBCDIC spaces, as in a SEG-Y textual header line
    std::vector<uint8_t> line(80, 0x40);
    const uint8_t text[] = {0xC3, 0x40, 0xF1, 0x40, 0xC3, 0xD3, 0xC9, 0xC5, 0xD5, 0xE3};
    std::copy(text, text + sizeof(text), line.begin());
    const std::string out = EbcdicConverter::to_ascii(line.data(), line.size());
    ASSERT_EQ(out.size(), 80u);
    EXPECT_EQ(out.substr(0, 10), "C 1 CLIENT");
    EXPECT_EQ(out.find_first_not_of(' ', 10), std::string::npos);
    EXPECT_EQ(EbcdicConverter::to_ascii(nullptr, 5), "");
    EXPECT_EQ(EbcdicConverter::to_ascii(line.data(), 0), "");
}

TEST(EbcdicTest, DetectsAsciiVersusEbcdicTextHeaders) {
    const std::string ascii_text =
        "C 1 CLIENT: ACME OIL            COMPANY: BASALT SURVEYS               DATE: 2024-01-01\n"
        "C 2 AREA: NORTH SEA BLOCK 22/4  CREW NO: 17                           LINE: 4711\n";
    EXPECT_TRUE(is_probably_ascii(reinterpret_cast<const uint8_t*>(ascii_text.data()), ascii_text.size()));

    // Same text encoded as EBCDIC (cp037)
    std::vector<uint8_t> ebcdic;
    for (char c : ascii_text) {
        if (c == '\n') { ebcdic.push_back(0x25); continue; }
        ebcdic.push_back(kCp037Printable[static_cast<size_t>(c - 0x20)]);
    }
    EXPECT_FALSE(is_probably_ascii(ebcdic.data(), ebcdic.size()));
    EXPECT_EQ(decode_text_header(ebcdic.data(), ebcdic.size()), ascii_text);
    EXPECT_EQ(decode_text_header(ascii_text.data(), ascii_text.size()), ascii_text);

    // Degenerate buffers
    EXPECT_FALSE(is_probably_ascii(nullptr, 10));
    EXPECT_FALSE(is_probably_ascii(ebcdic.data(), 0));
    const std::vector<uint8_t> ebcdic_blank(3200, 0x40);
    EXPECT_FALSE(is_probably_ascii(ebcdic_blank.data(), ebcdic_blank.size()));
    const std::vector<uint8_t> zeros(3200, 0x00);
    EXPECT_FALSE(is_probably_ascii(zeros.data(), zeros.size()));
    const std::vector<uint8_t> ascii_blank(3200, 0x20);
    EXPECT_TRUE(is_probably_ascii(ascii_blank.data(), ascii_blank.size()));
    EXPECT_EQ(decode_text_header(nullptr, 4), "");
}
