#pragma once

#include "basalt/segy_format.hpp"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace basalt::io {

// Thrown for every malformed, truncated or unsupported SEG-Y input. All
// functions in this header validate their input lengths and report problems
// only through this type (plus std::out_of_range for a bad trace index and
// std::bad_alloc), so they are safe to call on untrusted buffers.
class SegyFormatError : public std::runtime_error {
public:
    explicit SegyFormatError(const std::string& what) : std::runtime_error(what) {}
};

// ---------------------------------------------------------------------------
// Length-checked building blocks. 'size' is the number of readable bytes at
// 'buffer'; a buffer that is too short throws SegyFormatError.
// ---------------------------------------------------------------------------

// Reads the 3200-byte textual header and returns it as ASCII text. EBCDIC
// headers (the standard) are converted; headers that already look like ASCII
// (common in rev 2 files) are passed through unchanged.
// Requires size >= 3200.
std::string parse_text_header(const void* buffer, size_t size);

// Parses the 400-byte binary header. 'little_endian' selects the byte order of
// the file (rev 2 files may be little-endian); all multi-byte fields of the
// result are in host order. Requires size >= 400.
SegyBinaryHeader parse_binary_header(const void* buffer, size_t size, bool little_endian = false);

// Parses a 240-byte trace header; all fields of the result are in host order.
// Requires size >= 240.
SegyTraceHeader parse_trace_header(const void* buffer, size_t size, bool little_endian = false);

// ---------------------------------------------------------------------------
// Whole-file description
// ---------------------------------------------------------------------------

// Resolved geometry of a SEG-Y file. Produced by describe_segy(), which has
// already verified that every trace lies inside the buffer it was given.
struct SegyLayout {
    SegyBinaryHeader binary{};      // decoded binary header (host order)
    bool little_endian = false;     // byte order of all binary fields in the file
    uint8_t revision_major = 0;     // 0 for pre-rev-1 files
    uint8_t revision_minor = 0;
    bool fixed_length = true;       // fixed_length_flag from the header (informational)

    uint16_t data_sample_format = 0;  // validated code (1,2,3,5,6,7,8,9,10,11,12,15,16)
    size_t bytes_per_sample = 0;
    size_t samples_per_trace = 0;     // effective ns (> 0)

    size_t n_ext_text_headers = 0;    // number of 3200-byte extended textual headers
    size_t data_offset = 0;           // byte offset of the first trace header
    size_t trace_stride = 0;          // 240 + samples_per_trace * bytes_per_sample
    size_t n_traces = 0;
    size_t trailer_bytes = 0;         // rev 2 trailer stanzas after the last trace
};

// Validates the complete file and returns its layout.
//
// Checks performed (all arithmetic is overflow checked):
//  - size >= 3600 (text + binary header)
//  - byte order: a marker of 0x04030201 at binary offset 96 selects a
//    little-endian file; any other value means big-endian
//  - data_sample_format is a known SEG-Y code
//  - effective samples per trace is non-zero. The rev 2 extended field
//    (offset 68) overrides the 16-bit field when revision >= 2 and it is non-zero
//  - extended textual headers: count from offset 304 (revision >= 1). A count
//    of -1 (variable) is resolved by scanning for the "((SEG: EndText))"
//    stanza; if none is found before the end of the file, the file is rejected
//  - rev 2 files declaring additional trace headers are rejected (unsupported)
//  - size >= 3600 + n_ext * 3200 + trailer + n * (240 + ns * bytes_per_sample),
//    and the bytes after the header block divide evenly into whole traces; a
//    trailing partial trace is an error
//
// Variable-length traces are not supported: every trace is assumed to hold
// samples_per_trace samples regardless of fixed_length_flag, which is only
// reported (many writers leave it at 0 for fixed-length files).
SegyLayout describe_segy(const uint8_t* data, size_t size);

// Returns the 240-byte header of trace i (0-based). Throws std::out_of_range
// if i >= layout.n_traces and SegyFormatError if the layout does not fit 'size'.
SegyTraceHeader read_trace_header(const uint8_t* data, size_t size,
                                  const SegyLayout& layout, size_t i);

// Decodes the samples of trace i to float and writes layout.samples_per_trace
// values to 'out'. 'out_capacity' must be >= layout.samples_per_trace
// (std::invalid_argument otherwise). IBM float (1) is decoded via
// utils::ibm_to_ieee; IEEE float (5), integers (2,3,8,7,9,10,11,12,15,16) and
// IEEE double (6) are converted with the file's byte order.
void read_trace_samples(const uint8_t* data, size_t size, const SegyLayout& layout,
                        size_t i, float* out, size_t out_capacity);

// Convenience overload returning a vector of layout.samples_per_trace samples.
std::vector<float> read_trace_samples(const uint8_t* data, size_t size,
                                      const SegyLayout& layout, size_t i);

// Returns the number of bytes per sample for a data_sample_format code, or 0
// if the code is not a supported SEG-Y format.
size_t segy_bytes_per_sample(uint16_t data_sample_format);

} // namespace basalt::io
