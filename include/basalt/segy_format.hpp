#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <array>
#include <algorithm>
#include <type_traits>

namespace basalt {

constexpr size_t SEGY_TEXT_HEADER_SIZE = 3200;
constexpr size_t SEGY_BINARY_HEADER_SIZE = 400;
constexpr size_t SEGY_TRACE_HEADER_SIZE = 240;

// Host byte order, evaluated at compile time. Unknown compilers (e.g. MSVC)
// are assumed little-endian, which holds for every Windows target.
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && defined(__ORDER_LITTLE_ENDIAN__)
constexpr bool kHostIsLittleEndian = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
#else
constexpr bool kHostIsLittleEndian = true;
#endif

// Unconditional byte reversal of a 2, 4 or 8 byte integral value.
template <typename T>
[[nodiscard]] inline T byte_swap(T value) {
    static_assert(std::is_integral_v<T>, "byte_swap requires an integral type");
    static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "Only 16, 32 and 64 bit swaps supported");
    if constexpr (sizeof(T) == 2) {
        const uint16_t v = static_cast<uint16_t>(value);
        return static_cast<T>(static_cast<uint16_t>((v >> 8) | (v << 8)));
    } else if constexpr (sizeof(T) == 4) {
        const uint32_t v = static_cast<uint32_t>(value);
        return static_cast<T>(((v & 0xFF000000u) >> 24) |
                              ((v & 0x00FF0000u) >> 8)  |
                              ((v & 0x0000FF00u) << 8)  |
                              ((v & 0x000000FFu) << 24));
    } else {
        const uint64_t v = static_cast<uint64_t>(value);
        return static_cast<T>(((v & 0xFF00000000000000ull) >> 56) |
                              ((v & 0x00FF000000000000ull) >> 40) |
                              ((v & 0x0000FF0000000000ull) >> 24) |
                              ((v & 0x000000FF00000000ull) >> 8)  |
                              ((v & 0x00000000FF000000ull) << 8)  |
                              ((v & 0x0000000000FF0000ull) << 24) |
                              ((v & 0x000000000000FF00ull) << 40) |
                              ((v & 0x00000000000000FFull) << 56));
    }
}

// Converts between big-endian (the SEG-Y native order) and host order.
// The operation is its own inverse. It is a no-op on big-endian hosts.
template <typename T>
[[nodiscard]] inline T swap_endian(T value) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "Only 16, 32 and 64 bit swaps supported");
    if constexpr (kHostIsLittleEndian) {
        return byte_swap(value);
    } else {
        return value;
    }
}

// 400-byte Binary File Header, laid out exactly like the file (0-based offsets
// in comments; the SEG-Y spec numbers bytes from 3201). Multi-byte fields are
// host-endian after parse_binary_header() and file-endian before it.
struct SegyBinaryHeader {
    uint32_t job_id;                 // 0
    uint32_t line_number;            // 4
    uint32_t reel_number;            // 8
    uint16_t traces_per_ensemble;    // 12
    uint16_t aux_traces_per_ensemble;// 14
    uint16_t sample_interval;        // 16: microseconds (hdt)
    uint16_t sample_interval_orig;   // 18: microseconds (dto)
    uint16_t samples_per_trace;      // 20: ns
    uint16_t samples_per_trace_orig; // 22: nso
    uint16_t data_sample_format;     // 24: 1=IBM, 2=int32, 3=int16, 5=IEEE, 8=int8, ...
    uint16_t ensemble_fold;          // 26
    uint16_t trace_sorting_code;     // 28
    uint16_t vertical_sum_code;      // 30
    uint16_t sweep_frequency_start;  // 32
    uint16_t sweep_frequency_end;    // 34
    uint16_t sweep_length;           // 36
    uint16_t sweep_type;             // 38
    uint16_t trace_number_of_sweep_channel; // 40
    uint16_t sweep_trace_taper_length_start; // 42
    uint16_t sweep_trace_taper_length_end;   // 44
    uint16_t taper_type;             // 46
    uint16_t correlated_data_traces; // 48
    uint16_t binary_gain_recovered;  // 50
    uint16_t amplitude_recovery_method; // 52
    uint16_t measurement_system;     // 54: 1=Meters, 2=Feet
    uint16_t impulse_signal_polarity;// 56
    uint16_t vibratory_polarity_code;// 58

    // SEG-Y rev 2 extended fields (zero or unassigned in older files).
    uint32_t ext_traces_per_ensemble;     // 60
    uint32_t ext_aux_traces_per_ensemble; // 64
    uint32_t ext_samples_per_trace;       // 68: overrides samples_per_trace if non-zero (rev 2)
    double   ext_sample_interval;         // 72
    double   ext_sample_interval_orig;    // 80
    uint32_t ext_samples_per_trace_orig;  // 88
    uint32_t ext_ensemble_fold;           // 92
    uint32_t byte_order_marker;           // 96: 0x01020304 written in the file's own byte order
    uint8_t  unassigned_100[200];         // 100..299

    uint8_t  revision_major;              // 300 (bytes 3501): 8-bit, endian independent
    uint8_t  revision_minor;              // 301 (bytes 3502)
    int16_t  fixed_length_flag;           // 302: 1 = all traces have the same length
    int16_t  num_ext_text_headers;        // 304: 0 = none, -1 = variable (EndText terminated)
    uint8_t  max_additional_trace_headers_raw[4]; // 306: rev 2, raw bytes in file byte order
    int16_t  time_basis_code;             // 310
    uint64_t num_traces_in_file;          // 312: rev 2, 0 = unknown
    uint64_t first_trace_offset;          // 320: rev 2, 0 = unknown
    int32_t  num_trailer_stanzas;         // 328: rev 2, 3200-byte records after the last trace
    uint8_t  unassigned_332[68];          // 332..399
};

static_assert(sizeof(SegyBinaryHeader) == 400, "SegyBinaryHeader must be exactly 400 bytes");
static_assert(offsetof(SegyBinaryHeader, job_id) == 0, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, traces_per_ensemble) == 12, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, sample_interval) == 16, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, samples_per_trace) == 20, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, data_sample_format) == 24, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, trace_sorting_code) == 28, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, measurement_system) == 54, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, vibratory_polarity_code) == 58, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, ext_traces_per_ensemble) == 60, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, ext_samples_per_trace) == 68, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, ext_sample_interval) == 72, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, ext_ensemble_fold) == 92, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, byte_order_marker) == 96, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, revision_major) == 300, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, fixed_length_flag) == 302, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, num_ext_text_headers) == 304, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, max_additional_trace_headers_raw) == 306, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, num_traces_in_file) == 312, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, first_trace_offset) == 320, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, num_trailer_stanzas) == 328, "binary header layout");
static_assert(offsetof(SegyBinaryHeader, unassigned_332) == 332, "binary header layout");

// 240-byte Trace Header, laid out exactly like the file. Multi-byte fields are
// host-endian after parse_trace_header() and file-endian before it.
// Reference: SEG-Y Rev 1 specification (bytes 1-indexed in spec, 0-indexed here)
struct SegyTraceHeader {
    int32_t trace_seq_line;          // Bytes 1-4: Trace sequence number within line
    int32_t trace_seq_file;          // Bytes 5-8: Trace sequence number within file
    int32_t field_record_number;     // Bytes 9-12: Original field record number
    int32_t trace_number_field;      // Bytes 13-16: Trace number within field record
    int32_t energy_source_point;     // Bytes 17-20: Energy source point number
    int32_t cdp_ensemble_number;     // Bytes 21-24: CDP ensemble number
    int32_t trace_number_ensemble;   // Bytes 25-28: Trace number within ensemble
    int16_t trace_id_code;           // Bytes 29-30: Trace identification code
    int16_t num_vert_summed;         // Bytes 31-32
    int16_t num_horiz_stacked;       // Bytes 33-34
    int16_t data_use;                // Bytes 35-36
    int32_t source_receiver_offset;  // Bytes 37-40: Distance from source to receiver (KEY FIELD)
    int32_t receiver_elevation;      // Bytes 41-44
    int32_t source_elevation;        // Bytes 45-48
    int32_t source_depth;            // Bytes 49-52
    int32_t datum_elev_receiver;     // Bytes 53-56
    int32_t datum_elev_source;       // Bytes 57-60
    int32_t water_depth_source;      // Bytes 61-64
    int32_t water_depth_group;       // Bytes 65-68
    int16_t scalar_elev;             // Bytes 69-70: Scalar for elevations
    int16_t scalar_coord;            // Bytes 71-72: Scalar for coordinates
    int32_t source_x;                // Bytes 73-76
    int32_t source_y;                // Bytes 77-80
    int32_t group_x;                 // Bytes 81-84
    int32_t group_y;                 // Bytes 85-88
    int16_t coord_units;             // Bytes 89-90
    int16_t weathering_velocity;     // Bytes 91-92
    int16_t subweathering_velocity;  // Bytes 93-94
    int16_t uphole_time_source;      // Bytes 95-96
    int16_t uphole_time_group;       // Bytes 97-98
    int16_t source_static_corr;      // Bytes 99-100
    int16_t group_static_corr;       // Bytes 101-102
    int16_t total_static;            // Bytes 103-104
    int16_t lag_time_a;              // Bytes 105-106
    int16_t lag_time_b;              // Bytes 107-108
    int16_t delay_recording_time;    // Bytes 109-110: Delay recording time (KEY FIELD)
    int16_t mute_time_start;         // Bytes 111-112
    int16_t mute_time_end;           // Bytes 113-114
    uint16_t samples_this_trace;     // Bytes 115-116: Number of samples in this trace (KEY FIELD)
    uint16_t sample_interval_us;     // Bytes 117-118: Sample interval in microseconds

    // Bytes 119-180: gain, sweep, filter and time fields (31 x int16)
    int16_t gain_type;               // Bytes 119-120
    int16_t instrument_gain_const;   // Bytes 121-122
    int16_t instrument_early_gain;   // Bytes 123-124
    int16_t correlated;              // Bytes 125-126
    int16_t sweep_freq_start;        // Bytes 127-128
    int16_t sweep_freq_end;          // Bytes 129-130
    int16_t sweep_length;            // Bytes 131-132
    int16_t sweep_type;              // Bytes 133-134
    int16_t sweep_taper_start;       // Bytes 135-136
    int16_t sweep_taper_end;         // Bytes 137-138
    int16_t taper_type;              // Bytes 139-140
    int16_t alias_filter_freq;       // Bytes 141-142
    int16_t alias_filter_slope;      // Bytes 143-144
    int16_t notch_filter_freq;       // Bytes 145-146
    int16_t notch_filter_slope;      // Bytes 147-148
    int16_t low_cut_freq;            // Bytes 149-150
    int16_t high_cut_freq;           // Bytes 151-152
    int16_t low_cut_slope;           // Bytes 153-154
    int16_t high_cut_slope;          // Bytes 155-156
    int16_t year;                    // Bytes 157-158
    int16_t day_of_year;             // Bytes 159-160
    int16_t hour;                    // Bytes 161-162
    int16_t minute;                  // Bytes 163-164
    int16_t second;                  // Bytes 165-166
    int16_t time_basis_code;         // Bytes 167-168
    int16_t trace_weighting_factor;  // Bytes 169-170
    int16_t geophone_group_roll;     // Bytes 171-172
    int16_t geophone_group_first;    // Bytes 173-174
    int16_t geophone_group_last;     // Bytes 175-176
    int16_t gap_size;                // Bytes 177-178
    int16_t over_travel;             // Bytes 179-180

    // Bytes 181-240: SEG-Y rev 1 extended fields
    int32_t cdp_x;                   // Bytes 181-184
    int32_t cdp_y;                   // Bytes 185-188
    int32_t inline_number;           // Bytes 189-192
    int32_t crossline_number;        // Bytes 193-196
    int32_t shotpoint_number;        // Bytes 197-200
    int16_t scalar_shotpoint;        // Bytes 201-202
    int16_t trace_value_unit;        // Bytes 203-204
    int32_t transduction_mantissa;   // Bytes 205-208
    int16_t transduction_exponent;   // Bytes 209-210
    int16_t transduction_units;      // Bytes 211-212
    int16_t device_id;               // Bytes 213-214
    int16_t scalar_time;             // Bytes 215-216
    int16_t source_type;             // Bytes 217-218
    int16_t source_energy_direction[3]; // Bytes 219-224
    int32_t source_measurement_mantissa; // Bytes 225-228
    int16_t source_measurement_exponent; // Bytes 229-230
    int16_t source_measurement_unit; // Bytes 231-232
    uint8_t unassigned[8];           // Bytes 233-240
};

static_assert(sizeof(SegyTraceHeader) == 240, "SegyTraceHeader must be exactly 240 bytes");
static_assert(offsetof(SegyTraceHeader, trace_seq_line) == 0, "trace header layout");
static_assert(offsetof(SegyTraceHeader, trace_number_ensemble) == 24, "trace header layout");
static_assert(offsetof(SegyTraceHeader, trace_id_code) == 28, "trace header layout");
static_assert(offsetof(SegyTraceHeader, source_receiver_offset) == 36, "trace header layout");
static_assert(offsetof(SegyTraceHeader, water_depth_group) == 64, "trace header layout");
static_assert(offsetof(SegyTraceHeader, scalar_elev) == 68, "trace header layout");
static_assert(offsetof(SegyTraceHeader, source_x) == 72, "trace header layout");
static_assert(offsetof(SegyTraceHeader, group_y) == 84, "trace header layout");
static_assert(offsetof(SegyTraceHeader, coord_units) == 88, "trace header layout");
static_assert(offsetof(SegyTraceHeader, delay_recording_time) == 108, "trace header layout");
static_assert(offsetof(SegyTraceHeader, samples_this_trace) == 114, "trace header layout");
static_assert(offsetof(SegyTraceHeader, sample_interval_us) == 116, "trace header layout");
static_assert(offsetof(SegyTraceHeader, gain_type) == 118, "trace header layout");
static_assert(offsetof(SegyTraceHeader, alias_filter_freq) == 140, "trace header layout");
static_assert(offsetof(SegyTraceHeader, year) == 156, "trace header layout");
static_assert(offsetof(SegyTraceHeader, over_travel) == 178, "trace header layout");
static_assert(offsetof(SegyTraceHeader, cdp_x) == 180, "trace header layout");
static_assert(offsetof(SegyTraceHeader, cdp_y) == 184, "trace header layout");
static_assert(offsetof(SegyTraceHeader, inline_number) == 188, "trace header layout");
static_assert(offsetof(SegyTraceHeader, crossline_number) == 192, "trace header layout");
static_assert(offsetof(SegyTraceHeader, shotpoint_number) == 196, "trace header layout");
static_assert(offsetof(SegyTraceHeader, scalar_shotpoint) == 200, "trace header layout");
static_assert(offsetof(SegyTraceHeader, transduction_mantissa) == 204, "trace header layout");
static_assert(offsetof(SegyTraceHeader, scalar_time) == 214, "trace header layout");
static_assert(offsetof(SegyTraceHeader, source_type) == 216, "trace header layout");
static_assert(offsetof(SegyTraceHeader, source_energy_direction) == 218, "trace header layout");
static_assert(offsetof(SegyTraceHeader, source_measurement_mantissa) == 224, "trace header layout");
static_assert(offsetof(SegyTraceHeader, source_measurement_unit) == 230, "trace header layout");
static_assert(offsetof(SegyTraceHeader, unassigned) == 232, "trace header layout");

} // namespace basalt
