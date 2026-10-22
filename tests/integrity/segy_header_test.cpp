#include "basalt/io/segy_parser.hpp"
#include "basalt/segy_format.hpp"
#include "basalt/utils/ebcdic.hpp"
#include "basalt/utils/ibm_float.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace basalt;
using namespace basalt::io;

namespace {

using Bytes = std::vector<uint8_t>;

// Writes 'width' bytes of 'value' at 'off' in the requested byte order.
void put(Bytes& b, size_t off, uint64_t value, size_t width, bool le) {
    ASSERT_LE(off + width, b.size());
    for (size_t i = 0; i < width; ++i) {
        const size_t shift = 8 * (le ? i : width - 1 - i);
        b[off + i] = static_cast<uint8_t>((value >> shift) & 0xFF);
    }
}

// Reads 'width' bytes at 'off' in the requested byte order.
uint64_t get(const Bytes& b, size_t off, size_t width, bool le) {
    uint64_t v = 0;
    for (size_t i = 0; i < width; ++i) {
        const size_t shift = 8 * (le ? i : width - 1 - i);
        v |= static_cast<uint64_t>(b[off + i]) << shift;
    }
    return v;
}

struct Spec {
    uint16_t format = 5; // data_sample_format
    uint16_t ns = 4;     // 16-bit samples per trace
    uint32_t ext_ns = 0; // rev 2 extended samples per trace
    size_t traces = 3;
    size_t ext_records = 0; // extended textual header records actually present
    bool override_ext_field = false;
    int16_t ext_field = 0; // value written at offset 304 if override_ext_field
    uint8_t rev_major = 1;
    uint8_t rev_minor = 0;
    bool le = false;
    bool ascii_text = false;
    int32_t trailer_field = 0;     // rev 2 trailer stanza count written to the header
    size_t trailer_records = 0;    // trailer records actually appended
    uint32_t max_additional = 0;   // rev 2 additional trace headers
    uint32_t marker_override = 0;  // nonzero: raw value written (BE) at offset 96 instead of auto
    int endtext = 0;               // terminate the last ext record with EndText: 1 ASCII, 2 EBCDIC
    uint64_t num_traces_field = 0; // rev 2 number of traces in file (bytes 3513-3520), 0 = unknown
    uint64_t first_trace_field =
        0; // rev 2 byte offset of first trace (bytes 3521-3528), 0 = unknown
};

constexpr size_t kBin = 3200; // start of binary header
constexpr uint16_t kSupportedFormats[] = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16};
constexpr uint16_t kBadFormats[] = {0, 4, 13, 14, 17, 100, 255, 65535};
constexpr uint16_t kLeFormats[] = {1, 3, 5};

size_t bps_of(uint16_t fmt) {
    return io::segy_bytes_per_sample(fmt);
}

float sample_value(size_t t, size_t k) {
    return static_cast<float>(t * 10 + k) -
           7.0f; // small: fits every integer format, includes negatives
}

void encode_sample(Bytes& b, size_t off, uint16_t fmt, float v, bool le) {
    switch (fmt) {
    case 1:
        put(b, off, utils::ieee_to_ibm(v), 4, le);
        break;
    case 2:
        put(b, off, static_cast<uint32_t>(static_cast<int32_t>(v)), 4, le);
        break;
    case 3:
        put(b, off, static_cast<uint16_t>(static_cast<int16_t>(v)), 2, le);
        break;
    case 5: {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        put(b, off, u, 4, le);
        break;
    }
    case 6: {
        double d = v;
        uint64_t u;
        std::memcpy(&u, &d, 8);
        put(b, off, u, 8, le);
        break;
    }
    case 7:
        put(b, off, static_cast<uint32_t>(static_cast<int32_t>(v)) & 0xFFFFFFu, 3, le);
        break;
    case 8:
        put(b, off, static_cast<uint8_t>(static_cast<int8_t>(v)), 1, le);
        break;
    case 9:
        put(b, off, static_cast<uint64_t>(static_cast<int64_t>(v)), 8, le);
        break;
    case 10:
        put(b, off, static_cast<uint32_t>(v + 7.0f), 4, le);
        break;
    case 11:
        put(b, off, static_cast<uint16_t>(v + 7.0f), 2, le);
        break;
    case 12:
        put(b, off, static_cast<uint64_t>(v + 7.0f), 8, le);
        break;
    case 15:
        put(b, off, static_cast<uint32_t>(v + 7.0f), 3, le);
        break;
    case 16:
        put(b, off, static_cast<uint8_t>(v + 7.0f), 1, le);
        break;
    default:
        break;
    }
}

// Unsigned formats store v + 7 so that the decoded value is v + 7.
float expected_sample(uint16_t fmt, size_t t, size_t k) {
    const float v = sample_value(t, k);
    return (fmt == 10 || fmt == 11 || fmt == 12 || fmt == 15 || fmt == 16) ? v + 7.0f : v;
}

Bytes build_segy(const Spec& s) {
    const size_t eff_ns = (s.rev_major >= 2 && s.ext_ns != 0) ? s.ext_ns : s.ns;
    const size_t bps = bps_of(s.format) ? bps_of(s.format) : 4;
    const size_t stride = 240 + eff_ns * bps;
    Bytes b(3600 + s.ext_records * 3200 + s.traces * stride + s.trailer_records * 3200, 0);

    // Text header
    if (s.ascii_text) {
        std::fill(b.begin(), b.begin() + 3200, static_cast<uint8_t>(' '));
        const char* msg = "C 1 ASCII TEXT HEADER SURVEY";
        std::memcpy(b.data(), msg, std::strlen(msg));
    } else {
        std::fill(b.begin(), b.begin() + 3200, static_cast<uint8_t>(0x40));
        const uint8_t ebc[] = {0xC3, 0x40, 0xF1, 0x40, 0xC3, 0xD3, 0xC9, 0xC5, 0xD5, 0xE3};
        std::memcpy(b.data(), ebc, sizeof ebc);
    }

    // Binary header
    put(b, kBin + 16, 2000, 2, s.le); // hdt
    put(b, kBin + 20, s.ns, 2, s.le); // ns
    put(b, kBin + 24, s.format, 2, s.le);
    put(b, kBin + 68, s.ext_ns, 4, s.le);
    const uint32_t marker = s.marker_override
                                ? s.marker_override
                                : (s.le ? 0x04030201u : (s.rev_major >= 2 ? 0x01020304u : 0u));
    put(b, kBin + 96, marker, 4, false); // raw bytes: marker is "written in the file's own order"
    b[kBin + 300] = s.rev_major;
    b[kBin + 301] = s.rev_minor;
    put(b, kBin + 302, 1, 2, s.le);
    put(b, kBin + 304,
        static_cast<uint16_t>(s.override_ext_field ? s.ext_field
                                                   : static_cast<int16_t>(s.ext_records)),
        2, s.le);
    put(b, kBin + 306, s.max_additional, 4, s.le);
    put(b, kBin + 312, s.num_traces_field, 8, s.le);
    put(b, kBin + 320, s.first_trace_field, 8, s.le);
    put(b, kBin + 328, static_cast<uint32_t>(s.trailer_field), 4, s.le);

    if (s.endtext != 0 && s.ext_records > 0) {
        const char* ascii = "((SEG: EndText))";
        const uint8_t ebcdic[] = {0x4D, 0x4D, 0xE2, 0xC5, 0xC7, 0x7A, 0x40, 0xC5,
                                  0x95, 0x84, 0xE3, 0x85, 0xA7, 0xA3, 0x5D, 0x5D};
        uint8_t* dst = b.data() + 3600 + (s.ext_records - 1) * 3200 + 100;
        if (s.endtext == 1) {
            std::memcpy(dst, ascii, 16);
        } else {
            std::memcpy(dst, ebcdic, 16);
        }
    }

    // Traces
    const size_t base = 3600 + s.ext_records * 3200;
    for (size_t t = 0; t < s.traces; ++t) {
        const size_t off = base + t * stride;
        put(b, off + 4, t + 1, 4, s.le);    // trace_seq_file
        put(b, off + 114, eff_ns, 2, s.le); // samples_this_trace
        for (size_t k = 0; k < eff_ns; ++k) {
            encode_sample(b, off + 240 + k * bps, s.format, sample_value(t, k), s.le);
        }
    }
    return b;
}

} // namespace

TEST(SegyHeaderTest, SwapEndian) {
    uint16_t v16 = 0xAABB;
    EXPECT_EQ(byte_swap(v16), 0xBBAA);

    uint32_t v32 = 0xAABBCCDD;
    EXPECT_EQ(byte_swap(v32), 0xDDCCBBAAu);

    uint64_t v64 = 0x0102030405060708ull;
    EXPECT_EQ(byte_swap(v64), 0x0807060504030201ull);

    // swap_endian converts to/from big endian: identity on a big-endian host.
    EXPECT_EQ(swap_endian(v16), kHostIsLittleEndian ? static_cast<uint16_t>(0xBBAA) : v16);
    EXPECT_EQ(swap_endian(v32), kHostIsLittleEndian ? 0xDDCCBBAAu : v32);
    EXPECT_EQ(swap_endian(swap_endian(v64)), v64);

    int16_t neg = -2;
    EXPECT_EQ(byte_swap(byte_swap(neg)), neg);
}

TEST(SegyHeaderTest, EbcdicConversion) {
    // EBCDIC for "C01 CLIENT"
    // C=0xC3, 0=0xF0, 1=0xF1, space=0x40, C=0xC3, L=0xD3, I=0xC9, E=0xC5, N=0xD5, T=0xE3
    const unsigned char ebcdic_bytes[] = {0xC3, 0xF0, 0xF1, 0x40, 0xC3,
                                          0xD3, 0xC9, 0xC5, 0xD5, 0xE3};

    std::string ascii = utils::EbcdicConverter::to_ascii(ebcdic_bytes, sizeof(ebcdic_bytes));
    EXPECT_EQ(ascii, "C01 CLIENT");
}

TEST(SegyHeaderTest, TextHeaderEbcdicAndTruncation) {
    Bytes file = build_segy(Spec{});
    std::string text = parse_text_header(file.data(), file.size());
    ASSERT_EQ(text.size(), 3200u);
    EXPECT_EQ(text.substr(0, 10), "C 1 CLIENT");

    EXPECT_THROW(parse_text_header(file.data(), 3199), SegyFormatError);
    EXPECT_THROW(parse_text_header(file.data(), 0), SegyFormatError);
    EXPECT_THROW(parse_text_header(nullptr, 3200), SegyFormatError);
}

TEST(SegyHeaderTest, TextHeaderAsciiPassThrough) {
    Spec s;
    s.ascii_text = true;
    Bytes file = build_segy(s);
    std::string text = parse_text_header(file.data(), file.size());
    ASSERT_EQ(text.size(), 3200u);
    EXPECT_EQ(text.substr(0, 28), "C 1 ASCII TEXT HEADER SURVEY");
    EXPECT_EQ(text[100], ' ');

    // An ASCII header with NUL padding still passes through unchanged.
    Bytes raw(3200, 0);
    const char* msg = "C01 SURVEY NAME  C02 LINE 12  C03 CLIENT";
    std::memcpy(raw.data(), msg, std::strlen(msg));
    std::string t2 = parse_text_header(raw.data(), raw.size());
    EXPECT_EQ(t2.substr(0, std::strlen(msg)), msg);
    EXPECT_EQ(t2[3199], '\0');

    // All-zero header (neither encoding) must not crash.
    Bytes zeros(3200, 0);
    EXPECT_EQ(parse_text_header(zeros.data(), zeros.size()).size(), 3200u);
}

TEST(SegyHeaderTest, ParseBinaryHeader) {
    Bytes raw(400, 0);
    put(raw, 4, 101, 4, false);   // line_number
    put(raw, 16, 2000, 2, false); // sample_interval
    put(raw, 20, 1500, 2, false); // samples_per_trace
    put(raw, 24, 5, 2, false);    // IEEE Float

    SegyBinaryHeader parsed = parse_binary_header(raw.data(), raw.size());

    EXPECT_EQ(parsed.line_number, 101u);
    EXPECT_EQ(parsed.sample_interval, 2000);
    EXPECT_EQ(parsed.samples_per_trace, 1500);
    EXPECT_EQ(parsed.data_sample_format, 5);

    // Unaligned source buffer
    Bytes shifted(401, 0);
    std::memcpy(shifted.data() + 1, raw.data(), 400);
    SegyBinaryHeader p2 = parse_binary_header(shifted.data() + 1, 400);
    EXPECT_EQ(p2.samples_per_trace, 1500);
}

TEST(SegyHeaderTest, BinaryHeaderTruncatedAtEveryLength) {
    Bytes raw(400, 0);
    for (size_t n = 0; n < 400; ++n) {
        EXPECT_THROW(parse_binary_header(raw.data(), n), SegyFormatError) << "n=" << n;
    }
    EXPECT_NO_THROW(parse_binary_header(raw.data(), 400));
    EXPECT_THROW(parse_binary_header(nullptr, 400), SegyFormatError);
}

TEST(SegyHeaderTest, BinaryHeaderAllFieldsBothEndians) {
    // Each field's expected value is read straight from the bytes using the
    // SEG-Y rev 1/2 byte numbers (1-based byte N == offset N-1 - 3200).
    Bytes raw(400, 0);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = static_cast<uint8_t>(i * 37 + 11);
    }
    for (bool le : {false, true}) {
        SCOPED_TRACE(le ? "little" : "big");
        SegyBinaryHeader h = parse_binary_header(raw.data(), raw.size(), le);
        auto c16 = [&](size_t off, uint16_t v) {
            EXPECT_EQ(v, get(raw, off, 2, le)) << "off " << off;
        };
        auto c32 = [&](size_t off, uint32_t v) {
            EXPECT_EQ(v, get(raw, off, 4, le)) << "off " << off;
        };
        c32(0, h.job_id);
        c32(4, h.line_number);
        c32(8, h.reel_number);
        c16(12, h.traces_per_ensemble);
        c16(14, h.aux_traces_per_ensemble);
        c16(16, h.sample_interval);
        c16(18, h.sample_interval_orig);
        c16(20, h.samples_per_trace);
        c16(22, h.samples_per_trace_orig);
        c16(24, h.data_sample_format);
        c16(26, h.ensemble_fold);
        c16(28, h.trace_sorting_code);
        c16(30, h.vertical_sum_code);
        c16(32, h.sweep_frequency_start);
        c16(34, h.sweep_frequency_end);
        c16(36, h.sweep_length);
        c16(38, h.sweep_type);
        c16(40, h.trace_number_of_sweep_channel);
        c16(42, h.sweep_trace_taper_length_start);
        c16(44, h.sweep_trace_taper_length_end);
        c16(46, h.taper_type);
        c16(48, h.correlated_data_traces);
        c16(50, h.binary_gain_recovered);
        c16(52, h.amplitude_recovery_method);
        c16(54, h.measurement_system);
        c16(56, h.impulse_signal_polarity);
        c16(58, h.vibratory_polarity_code);
        c32(60, h.ext_traces_per_ensemble);
        c32(64, h.ext_aux_traces_per_ensemble);
        c32(68, h.ext_samples_per_trace);
        {
            uint64_t bits = get(raw, 72, 8, le);
            double d;
            std::memcpy(&d, &bits, 8);
            uint64_t got;
            std::memcpy(&got, &h.ext_sample_interval, 8);
            EXPECT_EQ(got, bits);
            bits = get(raw, 80, 8, le);
            std::memcpy(&got, &h.ext_sample_interval_orig, 8);
            EXPECT_EQ(got, bits);
            (void)d;
        }
        c32(88, h.ext_samples_per_trace_orig);
        c32(92, h.ext_ensemble_fold);
        c32(96, h.byte_order_marker);
        EXPECT_EQ(h.revision_major, raw[300]);
        EXPECT_EQ(h.revision_minor, raw[301]);
        c16(302, static_cast<uint16_t>(h.fixed_length_flag));
        c16(304, static_cast<uint16_t>(h.num_ext_text_headers));
        EXPECT_EQ(0, std::memcmp(h.max_additional_trace_headers_raw, raw.data() + 306, 4));
        c16(310, static_cast<uint16_t>(h.time_basis_code));
        EXPECT_EQ(h.num_traces_in_file, get(raw, 312, 8, le));
        EXPECT_EQ(h.first_trace_offset, get(raw, 320, 8, le));
        c32(328, static_cast<uint32_t>(h.num_trailer_stanzas));
        EXPECT_EQ(0, std::memcmp(h.unassigned_332, raw.data() + 332, 68));
        EXPECT_EQ(0, std::memcmp(h.unassigned_100, raw.data() + 100, 200));
    }
}

TEST(SegyHeaderTest, ParseTraceHeader) {
    Bytes raw(240, 0);
    put(raw, 36, 1500, 4, false);  // source_receiver_offset
    put(raw, 108, 100, 2, false);  // delay_recording_time
    put(raw, 114, 2001, 2, false); // samples_this_trace
    put(raw, 28, 1, 2, false);     // trace_id_code

    SegyTraceHeader parsed = parse_trace_header(raw.data(), raw.size());

    EXPECT_EQ(parsed.source_receiver_offset, 1500);
    EXPECT_EQ(parsed.delay_recording_time, 100);
    EXPECT_EQ(parsed.samples_this_trace, 2001);
    EXPECT_EQ(parsed.trace_id_code, 1);
}

TEST(SegyHeaderTest, TraceHeaderTruncatedAtEveryLength) {
    Bytes raw(240, 0);
    for (size_t n = 0; n < 240; ++n) {
        EXPECT_THROW(parse_trace_header(raw.data(), n), SegyFormatError) << "n=" << n;
    }
    EXPECT_NO_THROW(parse_trace_header(raw.data(), 240));
    EXPECT_THROW(parse_trace_header(nullptr, 240), SegyFormatError);
}

TEST(SegyHeaderTest, TraceHeaderAllFieldsRoundTrip) {
    // Offsets below are the SEG-Y rev 1 byte numbers minus one.
    Bytes raw(240, 0);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = static_cast<uint8_t>(i * 29 + 5);
    }
    for (bool le : {false, true}) {
        SCOPED_TRACE(le ? "little" : "big");
        SegyTraceHeader h = parse_trace_header(raw.data(), raw.size(), le);
        auto c16 = [&](size_t off, auto v) {
            EXPECT_EQ(static_cast<uint16_t>(v), get(raw, off, 2, le)) << "off " << off;
        };
        auto c32 = [&](size_t off, auto v) {
            EXPECT_EQ(static_cast<uint32_t>(v), get(raw, off, 4, le)) << "off " << off;
        };
        c32(0, h.trace_seq_line);
        c32(4, h.trace_seq_file);
        c32(8, h.field_record_number);
        c32(12, h.trace_number_field);
        c32(16, h.energy_source_point);
        c32(20, h.cdp_ensemble_number);
        c32(24, h.trace_number_ensemble);
        c16(28, h.trace_id_code);
        c16(30, h.num_vert_summed);
        c16(32, h.num_horiz_stacked);
        c16(34, h.data_use);
        c32(36, h.source_receiver_offset);
        c32(40, h.receiver_elevation);
        c32(44, h.source_elevation);
        c32(48, h.source_depth);
        c32(52, h.datum_elev_receiver);
        c32(56, h.datum_elev_source);
        c32(60, h.water_depth_source);
        c32(64, h.water_depth_group);
        c16(68, h.scalar_elev);
        c16(70, h.scalar_coord);
        c32(72, h.source_x);
        c32(76, h.source_y);
        c32(80, h.group_x);
        c32(84, h.group_y);
        c16(88, h.coord_units);
        c16(90, h.weathering_velocity);
        c16(92, h.subweathering_velocity);
        c16(94, h.uphole_time_source);
        c16(96, h.uphole_time_group);
        c16(98, h.source_static_corr);
        c16(100, h.group_static_corr);
        c16(102, h.total_static);
        c16(104, h.lag_time_a);
        c16(106, h.lag_time_b);
        c16(108, h.delay_recording_time);
        c16(110, h.mute_time_start);
        c16(112, h.mute_time_end);
        c16(114, h.samples_this_trace);
        c16(116, h.sample_interval_us);
        // bytes 119-180
        c16(118, h.gain_type);
        c16(120, h.instrument_gain_const);
        c16(122, h.instrument_early_gain);
        c16(124, h.correlated);
        c16(126, h.sweep_freq_start);
        c16(128, h.sweep_freq_end);
        c16(130, h.sweep_length);
        c16(132, h.sweep_type);
        c16(134, h.sweep_taper_start);
        c16(136, h.sweep_taper_end);
        c16(138, h.taper_type);
        c16(140, h.alias_filter_freq);
        c16(142, h.alias_filter_slope);
        c16(144, h.notch_filter_freq);
        c16(146, h.notch_filter_slope);
        c16(148, h.low_cut_freq);
        c16(150, h.high_cut_freq);
        c16(152, h.low_cut_slope);
        c16(154, h.high_cut_slope);
        c16(156, h.year);
        c16(158, h.day_of_year);
        c16(160, h.hour);
        c16(162, h.minute);
        c16(164, h.second);
        c16(166, h.time_basis_code);
        c16(168, h.trace_weighting_factor);
        c16(170, h.geophone_group_roll);
        c16(172, h.geophone_group_first);
        c16(174, h.geophone_group_last);
        c16(176, h.gap_size);
        c16(178, h.over_travel);
        // bytes 181-240
        c32(180, h.cdp_x);
        c32(184, h.cdp_y);
        c32(188, h.inline_number);
        c32(192, h.crossline_number);
        c32(196, h.shotpoint_number);
        c16(200, h.scalar_shotpoint);
        c16(202, h.trace_value_unit);
        c32(204, h.transduction_mantissa);
        c16(208, h.transduction_exponent);
        c16(210, h.transduction_units);
        c16(212, h.device_id);
        c16(214, h.scalar_time);
        c16(216, h.source_type);
        c16(218, h.source_energy_direction[0]);
        c16(220, h.source_energy_direction[1]);
        c16(222, h.source_energy_direction[2]);
        c32(224, h.source_measurement_mantissa);
        c16(228, h.source_measurement_exponent);
        c16(230, h.source_measurement_unit);
        EXPECT_EQ(0, std::memcmp(h.unassigned, raw.data() + 232, 8));
    }
}

TEST(SegyHeaderTest, DescribeValidFileAllFormats) {
    for (uint16_t fmt : kSupportedFormats) {
        SCOPED_TRACE("format " + std::to_string(fmt));
        Spec s;
        s.format = fmt;
        s.ns = 6;
        s.traces = 4;
        Bytes file = build_segy(s);
        SegyLayout lay = describe_segy(file.data(), file.size());
        EXPECT_EQ(lay.n_traces, 4u);
        EXPECT_EQ(lay.samples_per_trace, 6u);
        EXPECT_EQ(lay.bytes_per_sample, bps_of(fmt));
        EXPECT_EQ(lay.data_offset, 3600u);
        EXPECT_EQ(lay.trace_stride, 240 + 6 * bps_of(fmt));
        EXPECT_FALSE(lay.little_endian);
        EXPECT_EQ(lay.binary.sample_interval, 2000);
        for (size_t t = 0; t < lay.n_traces; ++t) {
            SegyTraceHeader th = read_trace_header(file.data(), file.size(), lay, t);
            EXPECT_EQ(th.trace_seq_file, static_cast<int32_t>(t + 1));
            EXPECT_EQ(th.samples_this_trace, 6);
            std::vector<float> v = read_trace_samples(file.data(), file.size(), lay, t);
            ASSERT_EQ(v.size(), 6u);
            for (size_t k = 0; k < 6; ++k) {
                EXPECT_EQ(v[k], expected_sample(fmt, t, k)) << "t=" << t << " k=" << k;
            }
        }
    }
}

TEST(SegyHeaderTest, IbmAndIeeeDecodeAgree) {
    // 118.625 is 0x4276A000 in IBM float and 0x42ED4000 in IEEE.
    Spec ibm;
    ibm.format = 1;
    ibm.ns = 2;
    ibm.traces = 1;
    Bytes fi = build_segy(ibm);
    put(fi, 3600 + 240, 0x4276A000u, 4, false);
    put(fi, 3600 + 244, 0xC276A000u, 4, false); // -118.625

    Spec ieee = ibm;
    ieee.format = 5;
    Bytes ff = build_segy(ieee);
    put(ff, 3600 + 240, 0x42ED4000u, 4, false);
    put(ff, 3600 + 244, 0xC2ED4000u, 4, false);

    SegyLayout li = describe_segy(fi.data(), fi.size());
    SegyLayout lf = describe_segy(ff.data(), ff.size());
    std::vector<float> vi = read_trace_samples(fi.data(), fi.size(), li, 0);
    std::vector<float> vf = read_trace_samples(ff.data(), ff.size(), lf, 0);
    EXPECT_FLOAT_EQ(vi[0], 118.625f);
    EXPECT_FLOAT_EQ(vi[1], -118.625f);
    EXPECT_FLOAT_EQ(vf[0], 118.625f);
    EXPECT_FLOAT_EQ(vf[1], -118.625f);
}

TEST(SegyHeaderTest, TruncationAtEveryBoundary) {
    // 2 extended text headers, 3 traces of 5 float samples. For every prefix
    // length: lengths inside the header block or ending mid-trace must throw,
    // lengths ending exactly on a trace boundary are valid shorter files.
    Spec s;
    s.ns = 5;
    s.traces = 3;
    s.ext_records = 2;
    Bytes full = build_segy(s);
    const size_t data_offset = 3600 + 2 * 3200;
    const size_t stride = 240 + 5 * 4;
    ASSERT_EQ(full.size(), data_offset + 3 * stride);
    for (size_t n = 0; n <= full.size(); ++n) {
        if (n < data_offset) {
            EXPECT_THROW(describe_segy(full.data(), n), SegyFormatError) << "n=" << n;
        } else if ((n - data_offset) % stride != 0) {
            EXPECT_THROW(describe_segy(full.data(), n), SegyFormatError) << "n=" << n;
        } else {
            SegyLayout lay = describe_segy(full.data(), n);
            EXPECT_EQ(lay.n_traces, (n - data_offset) / stride) << "n=" << n;
        }
    }
}

TEST(SegyHeaderTest, TruncatedTextAndBinaryHeaderBoundaries) {
    Bytes full = build_segy(Spec{});
    for (size_t n :
         {size_t{0}, size_t{1}, size_t{3199}, size_t{3200}, size_t{3201}, size_t{3599}}) {
        EXPECT_THROW(describe_segy(full.data(), n), SegyFormatError) << "n=" << n;
    }
    EXPECT_NO_THROW(describe_segy(full.data(), 3600 + 3 * (240 + 16)));
    EXPECT_THROW(describe_segy(nullptr, 5000), SegyFormatError);
}

TEST(SegyHeaderTest, TrailingPartialTraceIsError) {
    Spec s;
    Bytes file = build_segy(s);
    file.push_back(0);
    try {
        describe_segy(file.data(), file.size());
        FAIL() << "expected SegyFormatError";
    } catch (const SegyFormatError& e) {
        EXPECT_NE(std::string(e.what()).find("partial trace"), std::string::npos);
    }
}

TEST(SegyHeaderTest, AbsurdSizesAreRejectedNotOverflowed) {
    // ns = 65535, double samples, plus a stray byte: stride is large but sane in 64 bits.
    {
        Spec s;
        s.format = 6;
        s.ns = 65535;
        s.traces = 0;
        Bytes file = build_segy(s);
        file.resize(file.size() + 100);
        EXPECT_THROW(describe_segy(file.data(), file.size()), SegyFormatError);
    }
    // Rev 2 extended ns of 0xFFFFFFFF with 8-byte samples: stride ~ 34 GB.
    {
        Spec s;
        s.rev_major = 2;
        s.format = 6;
        s.ns = 0;
        s.ext_ns = 0xFFFFFFFFu;
        s.traces = 0;
        Bytes file = build_segy(s);
        // No traces at all is a valid (empty) file.
        EXPECT_NO_THROW(describe_segy(file.data(), file.size()));
        // One stray trace worth of bytes cannot hold a single such trace.
        file.resize(file.size() + 4096);
        EXPECT_THROW(describe_segy(file.data(), file.size()), SegyFormatError);
    }
    // Maximum extended header count with a tiny file.
    {
        Spec s;
        s.override_ext_field = true;
        s.ext_field = 32767;
        Bytes file = build_segy(s);
        EXPECT_THROW(describe_segy(file.data(), file.size()), SegyFormatError);
    }
    // Huge trailer count (rev 2) with a tiny file.
    {
        Spec s;
        s.rev_major = 2;
        s.trailer_field = std::numeric_limits<int32_t>::max();
        Bytes file = build_segy(s);
        EXPECT_THROW(describe_segy(file.data(), file.size()), SegyFormatError);
    }
    // Invalid negative extended header count.
    {
        Spec s;
        s.override_ext_field = true;
        s.ext_field = -2;
        Bytes file = build_segy(s);
        EXPECT_THROW(describe_segy(file.data(), file.size()), SegyFormatError);
    }
}

TEST(SegyHeaderTest, UnknownFormatsAndZeroSamplesRejected) {
    for (uint16_t fmt : kBadFormats) {
        Spec s;
        s.format = fmt;
        s.traces = 0;
        Bytes file = build_segy(s);
        EXPECT_THROW(describe_segy(file.data(), file.size()), SegyFormatError) << "fmt " << fmt;
    }
    Spec z;
    z.ns = 0;
    z.traces = 0;
    Bytes file = build_segy(z);
    EXPECT_THROW(describe_segy(file.data(), file.size()), SegyFormatError);
    EXPECT_EQ(segy_bytes_per_sample(4), 0u);
    EXPECT_EQ(segy_bytes_per_sample(1), 4u);
}

TEST(SegyHeaderTest, Rev2ExtendedSamplesPerTrace) {
    // ns field zero, extended field used.
    Spec s;
    s.rev_major = 2;
    s.ns = 0;
    s.ext_ns = 7;
    s.traces = 2;
    Bytes file = build_segy(s);
    SegyLayout lay = describe_segy(file.data(), file.size());
    EXPECT_EQ(lay.samples_per_trace, 7u);
    EXPECT_EQ(lay.n_traces, 2u);

    // Extended field overrides a non-zero 16-bit field in rev 2.
    Spec o = s;
    o.ns = 3;
    Bytes fo = build_segy(o);
    EXPECT_EQ(describe_segy(fo.data(), fo.size()).samples_per_trace, 7u);

    // In rev 1 the extended bytes are unassigned and ignored; ns==0 is an error.
    Spec r1 = s;
    r1.rev_major = 1;
    r1.traces = 0;
    Bytes f1 = build_segy(r1);
    EXPECT_THROW(describe_segy(f1.data(), f1.size()), SegyFormatError);

    // Rev 1 file with garbage in the unassigned extended area uses the 16-bit field.
    Spec r1b;
    r1b.rev_major = 1;
    r1b.ns = 4;
    r1b.ext_ns = 123456;
    Bytes f1b = build_segy(r1b);
    EXPECT_EQ(describe_segy(f1b.data(), f1b.size()).samples_per_trace, 4u);
}

TEST(SegyHeaderTest, LittleEndianRev2ViaByteOrderMarker) {
    for (uint16_t fmt : kLeFormats) {
        SCOPED_TRACE("format " + std::to_string(fmt));
        Spec s;
        s.rev_major = 2;
        s.le = true;
        s.format = fmt;
        s.ns = 5;
        s.traces = 3;
        Bytes file = build_segy(s);
        SegyLayout lay = describe_segy(file.data(), file.size());
        EXPECT_TRUE(lay.little_endian);
        EXPECT_EQ(lay.revision_major, 2);
        EXPECT_EQ(lay.binary.byte_order_marker, 0x01020304u);
        EXPECT_EQ(lay.binary.sample_interval, 2000);
        EXPECT_EQ(lay.binary.data_sample_format, fmt);
        EXPECT_EQ(lay.n_traces, 3u);
        for (size_t t = 0; t < 3; ++t) {
            SegyTraceHeader th = read_trace_header(file.data(), file.size(), lay, t);
            EXPECT_EQ(th.trace_seq_file, static_cast<int32_t>(t + 1));
            EXPECT_EQ(th.samples_this_trace, 5);
            std::vector<float> v = read_trace_samples(file.data(), file.size(), lay, t);
            for (size_t k = 0; k < 5; ++k) {
                EXPECT_EQ(v[k], expected_sample(fmt, t, k));
            }
        }
    }

    // A big-endian rev 2 file with the 0x01020304 marker stays big endian.
    Spec be;
    be.rev_major = 2;
    Bytes fb = build_segy(be);
    EXPECT_FALSE(describe_segy(fb.data(), fb.size()).little_endian);

    // Garbage in the marker area of an old file does not switch byte order.
    Spec old;
    old.rev_major = 0;
    old.marker_override = 0x12345678u;
    Bytes fo = build_segy(old);
    SegyLayout lo = describe_segy(fo.data(), fo.size());
    EXPECT_FALSE(lo.little_endian);
    EXPECT_EQ(lo.samples_per_trace, 4u);
}

TEST(SegyHeaderTest, ExtendedTextHeaders) {
    Spec s;
    s.ext_records = 2;
    s.traces = 2;
    Bytes file = build_segy(s);
    SegyLayout lay = describe_segy(file.data(), file.size());
    EXPECT_EQ(lay.n_ext_text_headers, 2u);
    EXPECT_EQ(lay.data_offset, 3600u + 6400u);
    EXPECT_EQ(lay.n_traces, 2u);

    // Rev 0 files do not define the count: ignore the (possibly garbage) field.
    Spec r0;
    r0.rev_major = 0;
    r0.override_ext_field = true;
    r0.ext_field = 5;
    Bytes f0 = build_segy(r0);
    SegyLayout l0 = describe_segy(f0.data(), f0.size());
    EXPECT_EQ(l0.n_ext_text_headers, 0u);
    EXPECT_EQ(l0.data_offset, 3600u);
}

TEST(SegyHeaderTest, VariableExtendedTextHeadersTerminatedByEndText) {
    Spec s;
    s.rev_major = 2;
    s.override_ext_field = true;
    s.ext_field = -1;
    s.ext_records = 3;
    s.endtext = 1;
    s.traces = 2;
    Bytes file = build_segy(s);
    SegyLayout lay = describe_segy(file.data(), file.size());
    EXPECT_EQ(lay.n_ext_text_headers, 3u);
    EXPECT_EQ(lay.data_offset, 3600u + 3 * 3200u);
    EXPECT_EQ(lay.n_traces, 2u);

    // EBCDIC-encoded terminator.
    Spec e = s;
    e.endtext = 2;
    e.ext_records = 2;
    Bytes fe = build_segy(e);
    SegyLayout le = describe_segy(fe.data(), fe.size());
    EXPECT_EQ(le.n_ext_text_headers, 2u);
    EXPECT_EQ(le.n_traces, 2u);

    // Unterminated: no EndText anywhere.
    Spec u = s;
    u.endtext = 0;
    Bytes unterminated = build_segy(u);
    EXPECT_THROW(describe_segy(unterminated.data(), unterminated.size()), SegyFormatError);
}

TEST(SegyHeaderTest, Rev2TrailerAndAdditionalTraceHeaders) {
    Spec s;
    s.rev_major = 2;
    s.traces = 3;
    s.trailer_field = 2;
    s.trailer_records = 2;
    Bytes file = build_segy(s);
    SegyLayout lay = describe_segy(file.data(), file.size());
    EXPECT_EQ(lay.n_traces, 3u);
    EXPECT_EQ(lay.trailer_bytes, 6400u);

    // Trailer present but not declared: looks like a partial trace.
    Spec undeclared = s;
    undeclared.trailer_field = 0;
    undeclared.trailer_records = 1; // 3200 bytes is not a multiple of the 256-byte stride
    Bytes fu = build_segy(undeclared);
    EXPECT_THROW(describe_segy(fu.data(), fu.size()), SegyFormatError);

    // Declared but missing.
    Spec missing = s;
    missing.trailer_records = 0;
    Bytes fm = build_segy(missing);
    EXPECT_THROW(describe_segy(fm.data(), fm.size()), SegyFormatError);

    // Additional trace headers change the stride: unsupported.
    Spec extra;
    extra.rev_major = 2;
    extra.max_additional = 1;
    Bytes fe = build_segy(extra);
    EXPECT_THROW(describe_segy(fe.data(), fe.size()), SegyFormatError);
}

TEST(SegyHeaderTest, Rev2NumTracesInFileMatchesComputedCount) {
    Spec s;
    s.rev_major = 2;
    s.traces = 3;

    // Zero means unknown and is accepted.
    Bytes unknown = build_segy(s);
    EXPECT_EQ(describe_segy(unknown.data(), unknown.size()).n_traces, 3u);

    // Matching count is accepted.
    s.num_traces_field = 3;
    Bytes match = build_segy(s);
    EXPECT_EQ(describe_segy(match.data(), match.size()).n_traces, 3u);

    // Matching count also holds with extended headers and a trailer.
    Spec full = s;
    full.ext_records = 2;
    full.endtext = 1;
    full.trailer_field = 1;
    full.trailer_records = 1;
    Bytes fb = build_segy(full);
    EXPECT_EQ(describe_segy(fb.data(), fb.size()).n_traces, 3u);
}

TEST(SegyHeaderTest, Rev2NumTracesInFileMismatchThrows) {
    Spec s;
    s.rev_major = 2;
    s.traces = 3;
    s.num_traces_field = 5;
    Bytes file = build_segy(s);
    try {
        (void)describe_segy(file.data(), file.size());
        ADD_FAILURE() << "expected SegyFormatError for a trace count mismatch";
    } catch (const SegyFormatError& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("declares 5 traces"), std::string::npos) << msg;
        EXPECT_NE(msg.find("holds 3"), std::string::npos) << msg;
    }

    // Rev 1 does not define the field, so a garbage value is ignored.
    Spec r1 = s;
    r1.rev_major = 1;
    Bytes f1 = build_segy(r1);
    EXPECT_EQ(describe_segy(f1.data(), f1.size()).n_traces, 3u);
}

TEST(SegyHeaderTest, Rev2FirstTraceOffsetMatchesComputedStart) {
    Spec s;
    s.rev_major = 2;
    s.ext_records = 2;
    s.traces = 2;
    s.first_trace_field = 3600 + 2 * 3200; // same as the computed start
    Bytes file = build_segy(s);
    SegyLayout lay = describe_segy(file.data(), file.size());
    EXPECT_EQ(lay.data_offset, 3600u + 2u * 3200u);
    EXPECT_EQ(lay.n_traces, 2u);
}

TEST(SegyHeaderTest, Rev2FirstTraceOffsetOverridesComputedStart) {
    // Three traces with 64 bytes of filler between the binary header and the
    // first trace. The declared offset (3664) is the true data start.
    Spec s;
    s.rev_major = 2;
    s.traces = 3;
    s.num_traces_field = 3;
    const Bytes base = build_segy(s);
    constexpr size_t kPad = 64;
    Bytes file(base.begin(), base.begin() + 3600);
    file.insert(file.end(), kPad, static_cast<uint8_t>(0xEE));
    file.insert(file.end(), base.begin() + 3600, base.end());
    put(file, kBin + 320, 3600 + kPad, 8, false); // rev 2 field, big-endian file

    SegyLayout lay = describe_segy(file.data(), file.size());
    EXPECT_EQ(lay.data_offset, 3600u + kPad);
    EXPECT_EQ(lay.n_traces, 3u);
    for (size_t t = 0; t < 3; ++t) {
        SegyTraceHeader th = read_trace_header(file.data(), file.size(), lay, t);
        EXPECT_EQ(th.trace_seq_file, static_cast<int32_t>(t + 1));
        std::vector<float> v = read_trace_samples(file.data(), file.size(), lay, t);
        for (size_t k = 0; k < 4; ++k) {
            EXPECT_EQ(v[k], expected_sample(5, t, k));
        }
    }

    // Without the declared offset the padding looks like a partial trace.
    Bytes unset = file;
    put(unset, kBin + 320, 0, 8, false);
    EXPECT_THROW(describe_segy(unset.data(), unset.size()), SegyFormatError);

    // Offsets that fall inside the headers or beyond the buffer are rejected.
    Bytes inside = file;
    put(inside, kBin + 320, 3599, 8, false);
    EXPECT_THROW(describe_segy(inside.data(), inside.size()), SegyFormatError);
    Bytes beyond = file;
    put(beyond, kBin + 320, file.size() + 1, 8, false);
    EXPECT_THROW(describe_segy(beyond.data(), beyond.size()), SegyFormatError);

    // Rev 1 ignores the field entirely.
    Spec r1;
    r1.rev_major = 1;
    r1.first_trace_field = 3599;
    Bytes f1 = build_segy(r1);
    EXPECT_EQ(describe_segy(f1.data(), f1.size()).data_offset, 3600u);
}

TEST(SegyHeaderTest, AsciiTextHeaderFileDescribes) {
    Spec s;
    s.ascii_text = true;
    Bytes file = build_segy(s);
    EXPECT_NO_THROW(describe_segy(file.data(), file.size()));
    EXPECT_EQ(parse_text_header(file.data(), file.size()).substr(0, 12), "C 1 ASCII TE");
}

TEST(SegyHeaderTest, TraceAccessorsAreBoundsChecked) {
    Spec s;
    s.traces = 2;
    Bytes file = build_segy(s);
    SegyLayout lay = describe_segy(file.data(), file.size());

    EXPECT_THROW(read_trace_header(file.data(), file.size(), lay, 2), std::out_of_range);
    EXPECT_THROW(read_trace_samples(file.data(), file.size(), lay, 99), std::out_of_range);
    EXPECT_THROW(
        read_trace_header(file.data(), file.size(), lay, std::numeric_limits<size_t>::max()),
        std::out_of_range);

    // Layout from a bigger buffer used with a shorter one.
    EXPECT_THROW(read_trace_header(file.data(), file.size() - 1, lay, 1), SegyFormatError);
    EXPECT_THROW(read_trace_header(nullptr, file.size(), lay, 0), SegyFormatError);

    float small[2];
    EXPECT_THROW(read_trace_samples(file.data(), file.size(), lay, 0, small, 2),
                 std::invalid_argument);
    float ok[4];
    EXPECT_NO_THROW(read_trace_samples(file.data(), file.size(), lay, 0, ok, 4));

    // Forged layout with an overflowing stride/offset must not read out of bounds.
    SegyLayout forged = lay;
    forged.n_traces = std::numeric_limits<size_t>::max();
    forged.trace_stride = std::numeric_limits<size_t>::max() / 2;
    EXPECT_THROW(read_trace_header(file.data(), file.size(), forged, 5), SegyFormatError);
    SegyLayout forged2 = lay;
    forged2.samples_per_trace = 1000000;
    EXPECT_THROW(read_trace_header(file.data(), file.size(), forged2, 0), SegyFormatError);
}

TEST(SegyHeaderTest, FormatErrorIsRuntimeError) {
    try {
        Bytes tiny(10, 0);
        describe_segy(tiny.data(), tiny.size());
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("truncated"), std::string::npos);
    }
}

TEST(SegyHeaderTest, DeterministicFuzzNeverCrashes) {
    // Stand-in for a libFuzzer target: under ASan/UBSan every input must either
    // parse or throw SegyFormatError, and never read out of bounds.
    std::mt19937 rng(0xB45A17u);
    auto rnd = [&](size_t n) { return static_cast<size_t>(rng() % n); };

    auto exercise = [&](const Bytes& buf) {
        try {
            (void)parse_text_header(buf.data(), buf.size());
        } catch (const SegyFormatError&) {
        }
        try {
            (void)parse_binary_header(buf.data(), buf.size(), (rng() & 1) != 0);
        } catch (const SegyFormatError&) {
        }
        try {
            (void)parse_trace_header(buf.data(), buf.size(), (rng() & 1) != 0);
        } catch (const SegyFormatError&) {
        }
        try {
            SegyLayout lay = describe_segy(buf.data(), buf.size());
            // A successful parse must account for every byte.
            ASSERT_EQ(lay.data_offset + lay.n_traces * lay.trace_stride + lay.trailer_bytes,
                      buf.size());
            std::vector<float> v;
            const size_t limit = std::min<size_t>(lay.n_traces, 8);
            for (size_t t = 0; t < limit; ++t) {
                (void)read_trace_header(buf.data(), buf.size(), lay, t);
                v.resize(lay.samples_per_trace);
                read_trace_samples(buf.data(), buf.size(), lay, t, v.data(), v.size());
            }
        } catch (const SegyFormatError&) {
        }
    };

    // 1. Pure random bytes of random size.
    for (int iter = 0; iter < 2000; ++iter) {
        Bytes buf(rnd(8 * 1024 + 1));
        for (auto& x : buf) {
            x = static_cast<uint8_t>(rng());
        }
        exercise(buf);
    }

    // 2. Valid files with a few mutated bytes in the header area, random truncation.
    for (int iter = 0; iter < 2000; ++iter) {
        Spec s;
        s.format = kSupportedFormats[rnd(13)];
        s.ns = static_cast<uint16_t>(1 + rnd(20));
        s.traces = rnd(4);
        s.ext_records = rnd(3);
        s.rev_major = static_cast<uint8_t>(rnd(3));
        s.le = (rng() & 1) != 0;
        s.endtext = static_cast<int>(rnd(3));
        Bytes buf = build_segy(s);
        const size_t flips = 1 + rnd(6);
        for (size_t f = 0; f < flips; ++f) {
            const size_t pos = (rng() & 1) ? kBin + rnd(400) : rnd(buf.size());
            buf[pos] = static_cast<uint8_t>(rng());
        }
        if (rng() & 1) {
            buf.resize(rnd(buf.size() + 1));
        }
        exercise(buf);
    }
}
