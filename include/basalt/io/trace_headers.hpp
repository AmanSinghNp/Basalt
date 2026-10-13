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

// Values to store in trace headers, one vector per field with the names of TraceHeaders. A vector
// that is empty leaves its field as it is; otherwise it holds one value per trace. Integer fields
// take whole numbers. Coordinates and receiver_elevation take real values, which are stored as
// round(value / scale), where the scale comes from the field's scalar.
struct TraceHeaderValues {
    std::vector<double> trace_sequence;
    std::vector<double> field_record;
    std::vector<double> channel;
    std::vector<double> offset;
    std::vector<double> scalar_elev;
    std::vector<double> scalar_coord;
    std::vector<double> coord_units;
    std::vector<double> source_x;
    std::vector<double> source_y;
    std::vector<double> group_x;
    std::vector<double> group_y;
    std::vector<double> receiver_elevation;
    std::vector<double> delay_ms;
    std::vector<double> year;
    std::vector<double> day_of_year;
    std::vector<double> hour;
    std::vector<double> minute;
    std::vector<double> second;
};

// Writes `values` into the raw big-endian trace headers, one entry per trace. A scalar that is not
// given is chosen for each trace: -100 when a coordinate (or the elevation) of that trace is not a
// whole number, and 1 otherwise. A scalar is written when it is given or when a value it scales is
// given. Throws std::invalid_argument, naming the field, when a vector length differs from
// raw.size(), a value is not finite, an integer field is not a whole number, or a stored value does
// not fit its field. Lengths and finiteness are checked before any byte is written; if a later
// value fails, raw may already hold the values written for earlier traces.
void encode_trace_headers(std::vector<std::array<uint8_t, 240>>& raw,
                          const TraceHeaderValues& values);

} // namespace basalt::io
