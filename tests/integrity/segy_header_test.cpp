#include "basalt/io/segy_parser.hpp"
#include "basalt/utils/ebcdic.hpp"
#include "basalt/segy_format.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstring>

using namespace basalt;
using namespace basalt::io;

TEST(SegyHeaderTest, SwapEndian) {
    uint16_t v16 = 0xAABB;
    EXPECT_EQ(swap_endian(v16), 0xBBAA);

    uint32_t v32 = 0xAABBCCDD;
    EXPECT_EQ(swap_endian(v32), 0xDDCCBBAA);
}

TEST(SegyHeaderTest, EbcdicConversion) {
    // EBCDIC for "C01 CLIENT"
    // C=0xC3, 0=0xF0, 1=0xF1, space=0x40, C=0xC3, L=0xD3, I=0xC9, E=0xC5, N=0xD5, T=0xE3
    const unsigned char ebcdic_bytes[] = { 
        0xC3, 0xF0, 0xF1, 0x40, 
        0xC3, 0xD3, 0xC9, 0xC5, 0xD5, 0xE3 
    };
    
    std::string ascii = utils::EbcdicConverter::to_ascii(ebcdic_bytes, sizeof(ebcdic_bytes));
    EXPECT_EQ(ascii, "C01 CLIENT");
}

TEST(SegyHeaderTest, ParseBinaryHeader) {
    // Simulated Big-Endian Binary Header
    SegyBinaryHeader be_header;
    std::memset(&be_header, 0, sizeof(be_header));

    // Sample Interval: 2000 us (2ms) -> 0x07D0 -> BE: 0xD007 (wait, 0x07D0 is 2000. Big endian storage means byte 0 is 07, byte 1 is D0)
    // Actually, let's write logical values and swap them to simulate "reading from disk"
    
    // We want the PARSER to see Big Endian data.
    // So we must WRITE Big Endian data into the buffer.
    
    auto to_be_u16 = [](uint16_t v) { return swap_endian(v); };
    auto to_be_u32 = [](uint32_t v) { return swap_endian(v); };

    be_header.line_number = to_be_u32(101);
    be_header.sample_interval = to_be_u16(2000);
    be_header.samples_per_trace = to_be_u16(1500);
    be_header.data_sample_format = to_be_u16(5); // IEEE Float

    SegyBinaryHeader parsed = parse_binary_header(&be_header);

    EXPECT_EQ(parsed.line_number, 101);
    EXPECT_EQ(parsed.sample_interval, 2000);
    EXPECT_EQ(parsed.samples_per_trace, 1500);
    EXPECT_EQ(parsed.data_sample_format, 5);
}

TEST(SegyHeaderTest, ParseTraceHeader) {
    // Simulated Big-Endian Trace Header
    SegyTraceHeader be_header;
    std::memset(&be_header, 0, sizeof(be_header));

    auto to_be_i16 = [](int16_t v) { return swap_endian(v); };
    auto to_be_u16 = [](uint16_t v) { return swap_endian(v); };
    auto to_be_i32 = [](int32_t v) { return swap_endian(v); };

    // Set key fields in Big-Endian format
    be_header.source_receiver_offset = to_be_i32(1500);     // 1500 meters offset
    be_header.delay_recording_time = to_be_i16(100);        // 100 ms delay
    be_header.samples_this_trace = to_be_u16(2001);         // 2001 samples
    be_header.trace_id_code = to_be_i16(1);                 // 1 = Production data

    SegyTraceHeader parsed = parse_trace_header(&be_header);

    EXPECT_EQ(parsed.source_receiver_offset, 1500);
    EXPECT_EQ(parsed.delay_recording_time, 100);
    EXPECT_EQ(parsed.samples_this_trace, 2001);
    EXPECT_EQ(parsed.trace_id_code, 1);
}
