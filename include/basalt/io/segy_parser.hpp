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

// Reads the 3200-byte textual header and returns it as 7-bit ASCII text of the
// same length. EBCDIC headers (the standard) are converted with cp037; headers
// that already look like ASCII (common in rev 2 files) are passed through, with
// any byte above 0x7F replaced by '?'. Detection uses utils::is_probably_ascii().
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
//
// Two layouts exist. Fixed-length files (the common case) have n_traces equally
// sized traces of samples_per_trace samples at data_offset + i * trace_stride.
// Variable-length files (variable_length == true) store a different sample count
// in each trace header; describe_segy() walks them once and records the byte
// offset and sample count of every trace. Use trace_offset(i) and
// trace_samples(i), which work in both modes, instead of computing offsets from
// trace_stride.
struct SegyLayout {
    SegyBinaryHeader binary{};  // decoded binary header (host order)
    bool little_endian = false; // byte order of all binary fields in the file
    uint8_t revision_major = 0; // 0 for pre-rev-1 files
    uint8_t revision_minor = 0;
    bool fixed_length = true; // fixed_length_flag from the header (informational)

    uint16_t data_sample_format = 0; // validated code (1,2,3,5,6,7,8,9,10,11,12,15,16)
    size_t bytes_per_sample = 0;
    // Fixed mode: samples in every trace (> 0). Variable mode: the largest
    // per-trace sample count (0 only if every trace is empty), so a buffer of
    // this many floats holds any trace.
    size_t samples_per_trace = 0;

    size_t n_ext_text_headers = 0; // number of 3200-byte extended textual headers
    size_t data_offset = 0;        // byte offset of the first trace header
    // 240 + samples_per_trace * bytes_per_sample in fixed mode; 0 in variable
    // mode, where traces are not equally spaced.
    size_t trace_stride = 0;
    size_t n_traces = 0;
    size_t trailer_bytes = 0; // rev 2 trailer stanzas after the last trace

    // Variable-length support. variable_length is the resolved decision, not
    // the header flag: see describe_segy().
    bool variable_length = false;
    size_t min_samples_per_trace = 0;    // smallest per-trace count (== samples_per_trace if fixed)
    std::vector<uint64_t> trace_offsets; // variable mode: byte offset of each trace header
    std::vector<uint32_t> trace_sample_counts; // variable mode: samples in each trace

    // Samples in trace i. Throws std::out_of_range if i >= n_traces.
    [[nodiscard]] size_t trace_samples(size_t i) const {
        if (i >= n_traces) {
            throw std::out_of_range("SEG-Y trace index " + std::to_string(i) + " out of range");
        }
        if (!variable_length) {
            return samples_per_trace;
        }
        if (i >= trace_sample_counts.size()) {
            throw std::out_of_range("SEG-Y variable-length layout has no entry for trace " +
                                    std::to_string(i));
        }
        return trace_sample_counts[i];
    }

    // Byte offset of the header of trace i in the file. In fixed mode this is
    // data_offset + i * trace_stride and is not overflow checked here (the
    // read_trace_* functions do that). Throws std::out_of_range like trace_samples().
    [[nodiscard]] uint64_t trace_offset(size_t i) const {
        if (i >= n_traces) {
            throw std::out_of_range("SEG-Y trace index " + std::to_string(i) + " out of range");
        }
        if (!variable_length) {
            return static_cast<uint64_t>(data_offset) + static_cast<uint64_t>(i) * trace_stride;
        }
        if (i >= trace_offsets.size()) {
            throw std::out_of_range("SEG-Y variable-length layout has no entry for trace " +
                                    std::to_string(i));
        }
        return trace_offsets[i];
    }
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
//  - rev 2 first_trace_offset (bytes 3521-3528), when non-zero, is the data
//    start: it overrides the offset computed from the extended headers. It
//    must lie past the header records and inside the buffer
//  - rev 2 num_traces_in_file (bytes 3513-3520), when non-zero, must equal the
//    computed trace count, otherwise SegyFormatError reports both numbers
//  - size >= 3600 + n_ext * 3200 + trailer + n * (240 + ns * bytes_per_sample),
//    and the bytes after the header block divide evenly into whole traces; a
//    trailing partial trace is an error
//
// Variable-length traces (SEG-Y rev 1/2): the layout is variable only when
// revision >= 1, the binary header's fixed_length_flag is 0, and the fixed
// interpretation does not hold, because either the bytes after the headers are
// not a whole number of (240 + ns * bytes_per_sample) traces, or some trace
// header's "samples in this trace" field (bytes 115-116, non-zero) differs from
// the binary header ns. Many writers leave the flag at 0 for fixed-length
// files; those keep the fast fixed layout. A trace header sample count of 0 is
// treated as "unset" for this decision, so it never forces variable mode on a
// file whose size is consistent. If the variable walk fails but the fixed
// interpretation is size-consistent and every non-zero trace header count is the
// same constant (a stale per-trace field), the fixed layout is used; headers
// that differ from each other and fail the walk are reported as damage, so a
// truncated variable-length file is never silently read as fixed. A zero binary
// header ns with flag 0 (revision >= 1) selects variable mode directly.
// Rev 0 files and files with flag 1 are always fixed-length.
//
// In variable mode each trace's length is its 16-bit header field (the rev 2
// 32-bit extension needs additional trace headers, which are unsupported). The
// walk checks every step: a trace whose header or samples would extend past the
// end of the trace region, a trailing partial trace header, and a first trace
// header that reports 0 samples while the binary header ns is non-zero (the
// headers evidently do not carry sample counts) are all SegyFormatErrors. A
// trace with 0 samples elsewhere is legal. samples_per_trace becomes the
// maximum count; trace_stride is 0.
SegyLayout describe_segy(const uint8_t* data, size_t size);

// Returns the 240-byte header of trace i (0-based). Throws std::out_of_range
// if i >= layout.n_traces and SegyFormatError if the layout does not fit 'size'
// (works for fixed and variable-length layouts).
SegyTraceHeader read_trace_header(const uint8_t* data, size_t size, const SegyLayout& layout,
                                  size_t i);

// Decodes the samples of trace i to float and writes layout.trace_samples(i)
// values to 'out'. 'out_capacity' must be >= layout.trace_samples(i) (always
// satisfied by layout.samples_per_trace; std::invalid_argument otherwise). IBM float (1) is decoded
// via utils::ibm_to_ieee; IEEE float (5), integers (2,3,8,7,9,10,11,12,15,16) and IEEE double (6)
// are converted with the file's byte order.
void read_trace_samples(const uint8_t* data, size_t size, const SegyLayout& layout, size_t i,
                        float* out, size_t out_capacity);

// Convenience overload returning a vector of layout.trace_samples(i) samples.
std::vector<float> read_trace_samples(const uint8_t* data, size_t size, const SegyLayout& layout,
                                      size_t i);

// Returns the number of bytes per sample for a data_sample_format code, or 0
// if the code is not a supported SEG-Y format.
size_t segy_bytes_per_sample(uint16_t data_sample_format);

} // namespace basalt::io
