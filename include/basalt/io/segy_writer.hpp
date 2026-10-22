#pragma once

#include "basalt/io/segy_parser.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace basalt::io {

// Thrown for I/O failures and for misuse of the writer (wrong trace count,
// unsupported output request). Format problems of an input file are reported
// by the parser as SegyFormatError, not by this type.
class SegyWriteError : public std::runtime_error {
public:
    explicit SegyWriteError(const std::string& what) : std::runtime_error(what) {}
};

// Streaming SEG-Y writer that reproduces the structure of an existing file
// (described by a SegyLayout) while replacing the trace samples.
//
// What is preserved: the caller passes the original header block (3200-byte
// textual header, 400-byte binary header and all extended textual headers,
// i.e. the first layout.data_offset bytes of the source file) and the raw
// 240-byte header of every trace. They are copied verbatim, so headers are
// byte-for-byte identical to the source, in the source's byte order.
//
// Sample encoding: samples are given as host floats and encoded in the byte
// order of the layout (big-endian by default; little-endian when the source
// was a little-endian rev 2 file).
//   * data_sample_format 5 (IEEE float): written exactly, bit for bit.
//   * data_sample_format 1 (IBM float): converted with utils::ieee_to_ibm
//     (round to nearest even into the 21..24 bit IBM fraction; NaN and Inf
//     saturate; the sign of zero is lost).
//   * every other format (integers, IEEE double, ...): the samples are written
//     as IEEE float (format 5) and the format code in the copied binary header
//     is patched to 5. This is the only header byte pair the writer changes.
//     Use output_format() to see which one applies.
//
// Variable-length layouts (layout.variable_length) are preserved: trace i is
// written with layout.trace_samples(i) samples, so every output trace has the
// same length as the source trace and the raw headers (which carry the per-trace
// sample counts) stay valid. The output re-parses to the same per-trace counts.
//
// Atomicity: data is written to a temporary file next to 'path' and renamed
// over 'path' by commit(). If commit() is never called (including when an
// exception unwinds), the temporary file is removed and 'path' is untouched.
//
// Usage:
//   SegyWriter w(out_path, src, layout.data_offset, layout);
//   for each trace i: w.write_trace(raw_header_i, samples_i);
//   w.commit(trailer_ptr, layout.trailer_bytes);
class SegyWriter {
public:
    // 'header_block' must point to 'header_bytes' bytes and header_bytes must
    // equal layout.data_offset. Throws SegyWriteError if it does not, or if
    // the temporary file cannot be created.
    SegyWriter(const std::string& path, const uint8_t* header_block, size_t header_bytes,
               const SegyLayout& layout);
    ~SegyWriter();

    SegyWriter(const SegyWriter&) = delete;
    SegyWriter& operator=(const SegyWriter&) = delete;

    // Appends one trace: 'trace_header' is the raw 240-byte header (file byte
    // order), 'samples' holds layout.samples_per_trace host floats (fixed
    // layouts) or layout.trace_samples(traces_written()) host floats (variable
    // layouts; 'samples' may be null for an empty trace).
    void write_trace(const uint8_t* trace_header, const float* samples);

    // Finishes the file. 'trailer' must point to 'trailer_bytes' bytes and
    // trailer_bytes must equal layout.trailer_bytes (it may be null when that
    // is 0). Requires exactly layout.n_traces traces to have been written, so
    // that the result parses back with describe_segy(). Flushes, closes and
    // renames the temporary file over the destination.
    void commit(const uint8_t* trailer = nullptr, size_t trailer_bytes = 0);

    // The data_sample_format code written to the output (1 or 5).
    [[nodiscard]] uint16_t output_format() const { return m_format; }
    [[nodiscard]] size_t traces_written() const { return m_written; }

private:
    void write_bytes(const void* p, size_t n);
    void discard() noexcept;

    std::string m_path;
    std::string m_tmp_path;
    std::ofstream m_out;
    bool m_le = false;
    uint16_t m_format = 5;
    size_t m_ns = 0; // fixed: samples per trace; variable: the maximum
    bool m_variable = false;
    std::vector<uint32_t> m_counts; // variable layouts: samples of each trace
    size_t m_expected_traces = 0;
    size_t m_trailer_bytes = 0;
    size_t m_written = 0;
    bool m_done = false;
    std::vector<uint8_t> m_buf; // one encoded trace payload
};

// Convenience: writes a file shaped like the source ('src', 'src_size',
// 'layout' from describe_segy) with new samples. 'samples' holds
// layout.n_traces * layout.samples_per_trace floats in trace-major order
// (trace 0's samples first). For variable-length layouts samples_per_trace is
// the maximum count, each trace's row starts at i * samples_per_trace and only
// its first layout.trace_samples(i) values are used. Headers, extended headers and trailer stanzas
// are copied from the source. See SegyWriter for the encoding rules.
void write_segy_like(const std::string& path, const uint8_t* src, size_t src_size,
                     const SegyLayout& layout, const float* samples);

} // namespace basalt::io
