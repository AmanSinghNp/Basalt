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

} // namespace basalt
