#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::io {

// Per-trace header fields, decoded from the raw 240-byte trace headers. Every vector has one
// entry per trace. Comments give the 1-based SEG-Y bytes. Coordinates and elevations have their
// scalar applied; the scalars themselves are kept raw.
struct TraceHeaders {
    std::vector<int32_t> trace_sequence;     // bytes 5-8: trace sequence number within file
    std::vector<int32_t> field_record;       // bytes 9-12: original field record number
    std::vector<int32_t> channel;            // bytes 13-16: trace number within field record
    std::vector<int32_t> offset;             // bytes 37-40: source to receiver offset
    std::vector<int16_t> scalar_elev;        // bytes 69-70: raw elevation scalar
    std::vector<int16_t> scalar_coord;       // bytes 71-72: raw coordinate scalar
    std::vector<int16_t> coord_units;        // bytes 89-90: 1 length, 2 arcseconds,
                                             // 3 decimal degrees, 4 DMS
    std::vector<double> source_x;            // bytes 73-76, scalar_coord applied
    std::vector<double> source_y;            // bytes 77-80, scalar_coord applied
    std::vector<double> group_x;             // bytes 81-84, scalar_coord applied
    std::vector<double> group_y;             // bytes 85-88, scalar_coord applied
    std::vector<double> receiver_elevation;  // bytes 41-44, scalar_elev applied
    std::vector<int16_t> delay_ms;           // bytes 109-110: delay recording time
    std::vector<int16_t> year;               // bytes 157-158
    std::vector<int16_t> day_of_year;        // bytes 159-160
    std::vector<int16_t> hour;               // bytes 161-162
    std::vector<int16_t> minute;             // bytes 163-164
    std::vector<int16_t> second;             // bytes 165-166

    // Sets the length of every vector to n.
    void resize(size_t n);
};

// Decodes the fields above from raw big-endian trace headers. A scalar follows the SEG-Y rule:
// positive multiplies, negative divides by its absolute value, and zero means 1.
TraceHeaders decode_trace_headers(const std::vector<std::array<uint8_t, 240>>& raw);

// Reads only the file and trace headers of a SEG-Y file. Sample data is not read or decoded.
// Throws std::runtime_error (naming the file) on the same malformed input as read_segy_headers.
TraceHeaders read_trace_headers(const std::string& path);

} // namespace basalt::io
