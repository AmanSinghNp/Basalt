#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::io {

// SEG-Y sample encodings, as stored in binary header bytes 3225-3226.
enum class SampleFormat : uint16_t { IBM32 = 1, INT32 = 2, INT16 = 3, IEEE32 = 5, INT8 = 8 };

// In-memory SEG-Y file. Headers are kept as raw bytes so that a read followed by a write
// reproduces them exactly (apart from the format code). `samples` holds the decoded data.
struct SegyData {
    std::string text_header;                        // 3200 chars, ASCII (converted from EBCDIC if needed)
    std::array<uint8_t, 3200> raw_text_header{};    // original bytes, written back verbatim
    std::array<uint8_t, 400> raw_binary_header{};   // original bytes
    std::vector<std::array<uint8_t, 240>> raw_trace_headers;
    SampleFormat format = SampleFormat::IEEE32;
    size_t ntraces = 0;
    size_t nsamples = 0;
    double dt = 0.0;                                // seconds, from binary header bytes 3217-3218 (us)
    std::vector<float> samples;                     // trace-major ntraces * nsamples
};

// Reads a big-endian SEG-Y rev 0/1 file with fixed trace length. Supports formats 1, 2, 3, 5, 8.
// Honors the extended textual header count (bytes 3505-3506) by skipping those headers; they
// are not retained, so a write-back records an extended header count of zero.
// Throws std::runtime_error (naming the file) on missing or malformed input.
SegyData read_segy(const std::string& path);

// Writes a SEG-Y file. Text and trace headers are written back from the raw arrays and the
// binary header is written from its raw bytes, except that the format code is set to
// `out_format` and the extended textual header count is set to zero. Samples are encoded in
// `out_format`. Supports writing IBM32 and IEEE32; throws std::invalid_argument otherwise, or
// when the sample count does not match the raw headers. Throws std::runtime_error on I/O failure.
void write_segy(const std::string& path, const SegyData& data,
                SampleFormat out_format = SampleFormat::IEEE32);

// Builds a minimal valid SegyData from samples (used by tests, the CLI and Python).
// `dt` is stored with microsecond resolution. Supports IBM32 and IEEE32 only.
SegyData make_segy(const std::vector<float>& samples, size_t ntraces, size_t nsamples,
                   double dt, SampleFormat format = SampleFormat::IEEE32);

} // namespace basalt::io
