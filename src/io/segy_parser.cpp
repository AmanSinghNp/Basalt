#include "basalt/io/segy_parser.hpp"
#include "basalt/utils/ebcdic.hpp"
#include "basalt/utils/ibm_float.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

namespace basalt::io {

namespace {

constexpr size_t kFileHeaderBytes = SEGY_TEXT_HEADER_SIZE + SEGY_BINARY_HEADER_SIZE; // 3600
constexpr uint32_t kByteOrderMarkerLittle = 0x04030201u; // 0x01020304 as seen by a big-endian reader

[[noreturn]] void fail(const std::string& what) {
    throw SegyFormatError(what);
}

void require_bytes(const void* buffer, size_t size, size_t needed, const char* what) {
    if (size < needed) {
        fail(std::string("SEG-Y ") + what + " truncated: need " + std::to_string(needed) +
             " bytes, have " + std::to_string(size));
    }
    if (buffer == nullptr) {
        fail(std::string("SEG-Y ") + what + ": null buffer");
    }
}

// Overflow-checked unsigned arithmetic on 64-bit values.
uint64_t checked_mul(uint64_t a, uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        fail(std::string("SEG-Y size overflow computing ") + what);
    }
    return a * b;
}

uint64_t checked_add(uint64_t a, uint64_t b, const char* what) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        fail(std::string("SEG-Y size overflow computing ") + what);
    }
    return a + b;
}

size_t to_size_t(uint64_t v, const char* what) {
    if (v > std::numeric_limits<size_t>::max()) {
        fail(std::string("SEG-Y value too large for this platform: ") + what);
    }
    return static_cast<size_t>(v);
}

// --- Raw byte loaders (alignment and host-endian independent) -------------

uint16_t load_u16(const uint8_t* p, bool le) {
    return le ? static_cast<uint16_t>(p[0] | (p[1] << 8))
              : static_cast<uint16_t>((p[0] << 8) | p[1]);
}

uint32_t load_u32(const uint8_t* p, bool le) {
    const uint32_t b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3];
    return le ? (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
              : ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3);
}

uint64_t load_u64(const uint8_t* p, bool le) {
    const uint64_t lo = load_u32(p, le);
    const uint64_t hi = load_u32(p + 4, le);
    return le ? (lo | (hi << 32)) : ((lo << 32) | hi);
}

uint32_t load_u24(const uint8_t* p, bool le) {
    const uint32_t b0 = p[0], b1 = p[1], b2 = p[2];
    return le ? (b0 | (b1 << 8) | (b2 << 16)) : ((b0 << 16) | (b1 << 8) | b2);
}

float double_to_float(double d) {
    if (std::isnan(d)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    // Out-of-range double to float conversion is undefined, so saturate.
    if (d > static_cast<double>(FLT_MAX)) return std::numeric_limits<float>::infinity();
    if (d < -static_cast<double>(FLT_MAX)) return -std::numeric_limits<float>::infinity();
    return static_cast<float>(d);
}

// "((SEG: EndText))" in ASCII and in EBCDIC (code page 037).
constexpr uint8_t kEndTextAscii[] = {0x28, 0x28, 0x53, 0x45, 0x47, 0x3A, 0x20, 0x45,
                                     0x6E, 0x64, 0x54, 0x65, 0x78, 0x74, 0x29, 0x29};
constexpr uint8_t kEndTextEbcdic[] = {0x4D, 0x4D, 0xE2, 0xC5, 0xC7, 0x7A, 0x40, 0xC5,
                                      0x95, 0x84, 0xE3, 0x85, 0xA7, 0xA3, 0x5D, 0x5D};

bool contains_end_text(const uint8_t* p, size_t n) {
    const uint8_t* end = p + n;
    return std::search(p, end, std::begin(kEndTextAscii), std::end(kEndTextAscii)) != end ||
           std::search(p, end, std::begin(kEndTextEbcdic), std::end(kEndTextEbcdic)) != end;
}

// In-place conversion helper: swaps only when file order differs from host.
struct Fixer {
    bool swap;
    template <typename T>
    void operator()(T& v) const {
        if (swap) {
            v = byte_swap(v);
        }
    }
    void operator()(double& v) const {
        if (swap) {
            uint64_t bits;
            std::memcpy(&bits, &v, sizeof bits);
            bits = byte_swap(bits);
            std::memcpy(&v, &bits, sizeof bits);
        }
    }
};

} // namespace

size_t segy_bytes_per_sample(uint16_t fmt) {
    switch (fmt) {
        case 1:  return 4; // IBM float
        case 2:  return 4; // int32
        case 3:  return 2; // int16
        case 5:  return 4; // IEEE float
        case 6:  return 8; // IEEE double
        case 7:  return 3; // int24
        case 8:  return 1; // int8
        case 9:  return 8; // int64
        case 10: return 4; // uint32
        case 11: return 2; // uint16
        case 12: return 8; // uint64
        case 15: return 3; // uint24
        case 16: return 1; // uint8
        default: return 0; // 4 (obsolete fixed point with gain) and unknown codes
    }
}

std::string parse_text_header(const void* buffer, size_t size) {
    require_bytes(buffer, size, SEGY_TEXT_HEADER_SIZE, "textual header");
    return utils::decode_text_header(buffer, SEGY_TEXT_HEADER_SIZE);
}

SegyBinaryHeader parse_binary_header(const void* buffer, size_t size, bool little_endian) {
    require_bytes(buffer, size, sizeof(SegyBinaryHeader), "binary header");

    // Copy to struct to ensure alignment (safe for unaligned buffer)
    SegyBinaryHeader h;
    std::memcpy(&h, buffer, sizeof(SegyBinaryHeader));

    const Fixer fix{little_endian != kHostIsLittleEndian};
    fix(h.job_id);
    fix(h.line_number);
    fix(h.reel_number);
    fix(h.traces_per_ensemble);
    fix(h.aux_traces_per_ensemble);
    fix(h.sample_interval);
    fix(h.sample_interval_orig);
    fix(h.samples_per_trace);
    fix(h.samples_per_trace_orig);
    fix(h.data_sample_format);
    fix(h.ensemble_fold);
    fix(h.trace_sorting_code);
    fix(h.vertical_sum_code);
    fix(h.sweep_frequency_start);
    fix(h.sweep_frequency_end);
    fix(h.sweep_length);
    fix(h.sweep_type);
    fix(h.trace_number_of_sweep_channel);
    fix(h.sweep_trace_taper_length_start);
    fix(h.sweep_trace_taper_length_end);
    fix(h.taper_type);
    fix(h.correlated_data_traces);
    fix(h.binary_gain_recovered);
    fix(h.amplitude_recovery_method);
    fix(h.measurement_system);
    fix(h.impulse_signal_polarity);
    fix(h.vibratory_polarity_code);

    fix(h.ext_traces_per_ensemble);
    fix(h.ext_aux_traces_per_ensemble);
    fix(h.ext_samples_per_trace);
    fix(h.ext_sample_interval);
    fix(h.ext_sample_interval_orig);
    fix(h.ext_samples_per_trace_orig);
    fix(h.ext_ensemble_fold);
    fix(h.byte_order_marker);

    // revision_major/minor are single bytes: no swap.
    fix(h.fixed_length_flag);
    fix(h.num_ext_text_headers);
    fix(h.time_basis_code);
    fix(h.num_traces_in_file);
    fix(h.first_trace_offset);
    fix(h.num_trailer_stanzas);

    return h;
}

SegyTraceHeader parse_trace_header(const void* buffer, size_t size, bool little_endian) {
    require_bytes(buffer, size, sizeof(SegyTraceHeader), "trace header");

    SegyTraceHeader h;
    std::memcpy(&h, buffer, sizeof(SegyTraceHeader));

    const Fixer fix{little_endian != kHostIsLittleEndian};

    fix(h.trace_seq_line);
    fix(h.trace_seq_file);
    fix(h.field_record_number);
    fix(h.trace_number_field);
    fix(h.energy_source_point);
    fix(h.cdp_ensemble_number);
    fix(h.trace_number_ensemble);
    fix(h.trace_id_code);
    fix(h.num_vert_summed);
    fix(h.num_horiz_stacked);
    fix(h.data_use);
    fix(h.source_receiver_offset);
    fix(h.receiver_elevation);
    fix(h.source_elevation);
    fix(h.source_depth);
    fix(h.datum_elev_receiver);
    fix(h.datum_elev_source);
    fix(h.water_depth_source);
    fix(h.water_depth_group);
    fix(h.scalar_elev);
    fix(h.scalar_coord);
    fix(h.source_x);
    fix(h.source_y);
    fix(h.group_x);
    fix(h.group_y);
    fix(h.coord_units);
    fix(h.weathering_velocity);
    fix(h.subweathering_velocity);
    fix(h.uphole_time_source);
    fix(h.uphole_time_group);
    fix(h.source_static_corr);
    fix(h.group_static_corr);
    fix(h.total_static);
    fix(h.lag_time_a);
    fix(h.lag_time_b);
    fix(h.delay_recording_time);
    fix(h.mute_time_start);
    fix(h.mute_time_end);
    fix(h.samples_this_trace);
    fix(h.sample_interval_us);

    fix(h.gain_type);
    fix(h.instrument_gain_const);
    fix(h.instrument_early_gain);
    fix(h.correlated);
    fix(h.sweep_freq_start);
    fix(h.sweep_freq_end);
    fix(h.sweep_length);
    fix(h.sweep_type);
    fix(h.sweep_taper_start);
    fix(h.sweep_taper_end);
    fix(h.taper_type);
    fix(h.alias_filter_freq);
    fix(h.alias_filter_slope);
    fix(h.notch_filter_freq);
    fix(h.notch_filter_slope);
    fix(h.low_cut_freq);
    fix(h.high_cut_freq);
    fix(h.low_cut_slope);
    fix(h.high_cut_slope);
    fix(h.year);
    fix(h.day_of_year);
    fix(h.hour);
    fix(h.minute);
    fix(h.second);
    fix(h.time_basis_code);
    fix(h.trace_weighting_factor);
    fix(h.geophone_group_roll);
    fix(h.geophone_group_first);
    fix(h.geophone_group_last);
    fix(h.gap_size);
    fix(h.over_travel);

    fix(h.cdp_x);
    fix(h.cdp_y);
    fix(h.inline_number);
    fix(h.crossline_number);
    fix(h.shotpoint_number);
    fix(h.scalar_shotpoint);
    fix(h.trace_value_unit);
    fix(h.transduction_mantissa);
    fix(h.transduction_exponent);
    fix(h.transduction_units);
    fix(h.device_id);
    fix(h.scalar_time);
    fix(h.source_type);
    for (int16_t& d : h.source_energy_direction) {
        fix(d);
    }
    fix(h.source_measurement_mantissa);
    fix(h.source_measurement_exponent);
    fix(h.source_measurement_unit);

    return h;
}

SegyLayout describe_segy(const uint8_t* data, size_t size) {
    require_bytes(data, size, kFileHeaderBytes, "file header");

    SegyLayout lay;

    // Byte order marker: 0x01020304 stored in the file's own order. Only the
    // exact little-endian pattern switches the order, because rev 0/1 files
    // may carry arbitrary bytes in this unassigned area.
    const uint8_t* bin = data + SEGY_TEXT_HEADER_SIZE;
    lay.little_endian = load_u32(bin + offsetof(SegyBinaryHeader, byte_order_marker), false) ==
                        kByteOrderMarkerLittle;
    lay.binary = parse_binary_header(bin, SEGY_BINARY_HEADER_SIZE, lay.little_endian);
    const SegyBinaryHeader& b = lay.binary;

    lay.revision_major = b.revision_major;
    lay.revision_minor = b.revision_minor;
    lay.fixed_length = (b.fixed_length_flag != 0);

    // Sample format
    lay.data_sample_format = b.data_sample_format;
    lay.bytes_per_sample = segy_bytes_per_sample(b.data_sample_format);
    if (lay.bytes_per_sample == 0) {
        fail("SEG-Y unsupported data sample format code " + std::to_string(b.data_sample_format));
    }

    // Samples per trace; the rev 2 extended field overrides when present.
    uint64_t ns = b.samples_per_trace;
    if (lay.revision_major >= 2 && b.ext_samples_per_trace != 0) {
        ns = b.ext_samples_per_trace;
    }
    if (ns == 0) {
        fail("SEG-Y samples per trace is zero");
    }
    lay.samples_per_trace = to_size_t(ns, "samples per trace");

    // Rev 2 additional trace headers change the trace stride; not supported.
    if (lay.revision_major >= 2) {
        const uint32_t extra =
            load_u32(bin + offsetof(SegyBinaryHeader, max_additional_trace_headers_raw),
                     lay.little_endian);
        if (extra != 0) {
            fail("SEG-Y additional trace headers are not supported");
        }
    }

    // Extended textual headers (introduced in rev 1)
    uint64_t pos = kFileHeaderBytes;
    if (lay.revision_major >= 1) {
        const int16_t n_ext = b.num_ext_text_headers;
        if (n_ext >= 0) {
            lay.n_ext_text_headers = static_cast<size_t>(n_ext);
            pos = checked_add(pos, checked_mul(static_cast<uint64_t>(n_ext),
                                               SEGY_TEXT_HEADER_SIZE, "extended header bytes"),
                              "data offset");
            if (pos > size) {
                fail("SEG-Y extended textual headers truncated: need " + std::to_string(pos) +
                     " bytes, have " + std::to_string(size));
            }
        } else if (n_ext == -1) {
            // Variable number: records follow until one contains the EndText stanza.
            size_t count = 0;
            bool found = false;
            while (!found) {
                if (pos > size || size - static_cast<size_t>(pos) < SEGY_TEXT_HEADER_SIZE) {
                    fail("SEG-Y variable extended textual headers not terminated by EndText");
                }
                found = contains_end_text(data + pos, SEGY_TEXT_HEADER_SIZE);
                pos += SEGY_TEXT_HEADER_SIZE;
                ++count;
            }
            lay.n_ext_text_headers = count;
        } else {
            fail("SEG-Y invalid number of extended textual headers " + std::to_string(n_ext));
        }
    }
    lay.data_offset = to_size_t(pos, "data offset");

    // Rev 2 trailer stanzas follow the last trace.
    uint64_t trailer = 0;
    if (lay.revision_major >= 2 && b.num_trailer_stanzas > 0) {
        trailer = checked_mul(static_cast<uint64_t>(b.num_trailer_stanzas),
                              SEGY_TEXT_HEADER_SIZE, "trailer bytes");
    }

    const uint64_t stride = checked_add(
        SEGY_TRACE_HEADER_SIZE,
        checked_mul(ns, lay.bytes_per_sample, "trace payload bytes"), "trace stride");

    const uint64_t remaining = size - lay.data_offset;
    if (trailer > remaining) {
        fail("SEG-Y trailer stanzas truncated");
    }
    const uint64_t trace_region = remaining - trailer;
    if (trace_region % stride != 0) {
        fail("SEG-Y trailing partial trace: " + std::to_string(trace_region % stride) +
             " stray bytes after " + std::to_string(trace_region / stride) + " complete traces");
    }
    lay.trace_stride = to_size_t(stride, "trace stride");
    lay.n_traces = to_size_t(trace_region / stride, "trace count");
    lay.trailer_bytes = to_size_t(trailer, "trailer bytes");
    return lay;
}

namespace {

// Returns a pointer to the start of trace i after re-validating that the whole
// trace lies inside [data, data + size).
const uint8_t* locate_trace(const uint8_t* data, size_t size, const SegyLayout& lay, size_t i) {
    if (i >= lay.n_traces) {
        throw std::out_of_range("SEG-Y trace index " + std::to_string(i) + " out of range (" +
                                std::to_string(lay.n_traces) + " traces)");
    }
    if (data == nullptr) {
        fail("SEG-Y null buffer");
    }
    const uint64_t payload = checked_mul(lay.samples_per_trace, lay.bytes_per_sample, "payload");
    if (checked_add(SEGY_TRACE_HEADER_SIZE, payload, "trace size") > lay.trace_stride) {
        fail("SEG-Y layout inconsistent: trace stride smaller than a trace");
    }
    const uint64_t begin = checked_add(
        lay.data_offset, checked_mul(i, lay.trace_stride, "trace offset"), "trace offset");
    const uint64_t end = checked_add(begin, lay.trace_stride, "trace end");
    if (end > size) {
        fail("SEG-Y trace " + std::to_string(i) + " lies beyond the end of the buffer");
    }
    return data + begin;
}

} // namespace

SegyTraceHeader read_trace_header(const uint8_t* data, size_t size,
                                  const SegyLayout& layout, size_t i) {
    const uint8_t* p = locate_trace(data, size, layout, i);
    return parse_trace_header(p, SEGY_TRACE_HEADER_SIZE, layout.little_endian);
}

void read_trace_samples(const uint8_t* data, size_t size, const SegyLayout& layout,
                        size_t i, float* out, size_t out_capacity) {
    const uint8_t* p = locate_trace(data, size, layout, i);
    const size_t ns = layout.samples_per_trace;
    if (out_capacity < ns || (ns != 0 && out == nullptr)) {
        throw std::invalid_argument("SEG-Y output buffer too small for trace samples");
    }
    const bool le = layout.little_endian;
    const uint8_t* s = p + SEGY_TRACE_HEADER_SIZE;
    const size_t bps = layout.bytes_per_sample;

    switch (layout.data_sample_format) {
        case 1:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = utils::ibm_to_ieee(load_u32(s + k * bps, le));
            }
            break;
        case 2:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(static_cast<int32_t>(load_u32(s + k * bps, le)));
            }
            break;
        case 3:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(static_cast<int16_t>(load_u16(s + k * bps, le)));
            }
            break;
        case 5:
            for (size_t k = 0; k < ns; ++k) {
                const uint32_t bits = load_u32(s + k * bps, le);
                float f;
                std::memcpy(&f, &bits, sizeof f);
                out[k] = f;
            }
            break;
        case 6:
            for (size_t k = 0; k < ns; ++k) {
                const uint64_t bits = load_u64(s + k * bps, le);
                double d;
                std::memcpy(&d, &bits, sizeof d);
                out[k] = double_to_float(d);
            }
            break;
        case 7:
            for (size_t k = 0; k < ns; ++k) {
                const uint32_t v = load_u24(s + k * bps, le);
                out[k] = static_cast<float>(static_cast<int32_t>(v ^ 0x800000u) - 0x800000);
            }
            break;
        case 8:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(static_cast<int8_t>(s[k]));
            }
            break;
        case 9:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(static_cast<int64_t>(load_u64(s + k * bps, le)));
            }
            break;
        case 10:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(load_u32(s + k * bps, le));
            }
            break;
        case 11:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(load_u16(s + k * bps, le));
            }
            break;
        case 12:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(load_u64(s + k * bps, le));
            }
            break;
        case 15:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(load_u24(s + k * bps, le));
            }
            break;
        case 16:
            for (size_t k = 0; k < ns; ++k) {
                out[k] = static_cast<float>(s[k]);
            }
            break;
        default:
            fail("SEG-Y unsupported data sample format code " +
                 std::to_string(layout.data_sample_format));
    }
}

std::vector<float> read_trace_samples(const uint8_t* data, size_t size,
                                      const SegyLayout& layout, size_t i) {
    std::vector<float> out(layout.samples_per_trace);
    read_trace_samples(data, size, layout, i, out.data(), out.size());
    return out;
}

} // namespace basalt::io
