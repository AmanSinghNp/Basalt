#pragma once

#include <cstdint>
#include <array>
#include <algorithm>

namespace basalt {

constexpr size_t SEGY_TEXT_HEADER_SIZE = 3200;
constexpr size_t SEGY_BINARY_HEADER_SIZE = 400;

// Endian swapping utility (BSWAP wrapper)
template <typename T>
[[nodiscard]] inline T swap_endian(T value) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "Only 16 and 32 bit swaps supported");
    if constexpr (sizeof(T) == 2) {
        return static_cast<T>((static_cast<uint16_t>(value) >> 8) | 
                              (static_cast<uint16_t>(value) << 8));
    } else {
        uint32_t v = static_cast<uint32_t>(value);
        return static_cast<T>(((v & 0xFF000000) >> 24) |
                              ((v & 0x00FF0000) >> 8)  |
                              ((v & 0x0000FF00) << 8)  |
                              ((v & 0x000000FF) << 24));
    }
}

// 400-byte Binary File Header
// Note: We use uint8_t arrays for padding to ensure exact 400 byte size layout
// Fields are usually Big-Endian in the file.
struct SegyBinaryHeader {
    uint32_t job_id;
    uint32_t line_number;
    uint32_t reel_number;
    uint16_t traces_per_ensemble;
    uint16_t aux_traces_per_ensemble;
    uint16_t sample_interval;        // microseconds (hdt)
    uint16_t sample_interval_orig;   // microseconds (dto)
    uint16_t samples_per_trace;      // ns
    uint16_t samples_per_trace_orig; // nso
    uint16_t data_sample_format;     // 1=IBM Float, 5=IEEE Float
    uint16_t ensemble_fold;
    uint16_t trace_sorting_code;
    uint16_t vertical_sum_code;
    uint16_t sweep_frequency_start;
    uint16_t sweep_frequency_end;
    uint16_t sweep_length;
    uint16_t sweep_type;
    uint16_t trace_number_of_sweep_channel;
    uint16_t sweep_trace_taper_length_start;
    uint16_t sweep_trace_taper_length_end;
    uint16_t taper_type;
    uint16_t correlated_data_traces;
    uint16_t binary_gain_recovered;
    uint16_t amplitude_recovery_method;
    uint16_t measurement_system;     // 1=Meters, 2=Feet
    uint16_t impulse_signal_polarity;
    uint16_t vibratory_polarity_code;
    
    // Unassigned (240 bytes) + Segy Revision info (60 bytes)
    // To keep it simple and aligned, we padding out to 400 bytes.
    // The explicit fields above take 4 * 3 + 2 * 24 = 12 + 48 = 60 bytes.
    // We need 340 bytes of padding.
    uint8_t padding[340];
};

static_assert(sizeof(SegyBinaryHeader) == 400, "SegyBinaryHeader must be exactly 400 bytes");

// 240-byte Trace Header
// Fields are Big-Endian in the file.
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
    // Bytes 119-180: Various gain/filter fields (62 bytes)
    uint8_t gain_and_filter[62];
    // Bytes 181-240: Reserved and unassigned (60 bytes)
    uint8_t reserved[60];
};

static_assert(sizeof(SegyTraceHeader) == 240, "SegyTraceHeader must be exactly 240 bytes");

} // namespace basalt
