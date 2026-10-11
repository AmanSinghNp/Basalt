#include "basalt/io/segy_file.hpp"
#include "basalt/io/segy_parser.hpp"
#include "basalt/segy_format.hpp"
#include "basalt/utils/ibm_float.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace basalt::io {

namespace {

// Layout constants. Offsets are 0-based within each header; the 1-based standard byte
// numbers are given in the comments (file offset = 3200 + (byte - 3201) for binary fields).
constexpr size_t kTextHeaderSize = SEGY_TEXT_HEADER_SIZE;       // 3200
constexpr size_t kBinaryHeaderSize = SEGY_BINARY_HEADER_SIZE;   // 400
constexpr size_t kFileHeaderSize = kTextHeaderSize + kBinaryHeaderSize;  // 3600
constexpr size_t kExtTextHeaderSize = 3200;
constexpr size_t kTraceHeaderSize = 240;

constexpr size_t kBinSampleIntervalOff = 16;   // bytes 3217-3218
constexpr size_t kBinSamplesOff = 20;          // bytes 3221-3222
constexpr size_t kBinFormatOff = 24;           // bytes 3225-3226
constexpr size_t kBinRevisionOff = 300;        // bytes 3501-3502
constexpr size_t kBinFixedLengthOff = 302;     // bytes 3503-3504
constexpr size_t kBinExtCountOff = 304;        // bytes 3505-3506

constexpr size_t kTrSeqLineOff = 0;            // bytes 1-4
constexpr size_t kTrSeqFileOff = 4;            // bytes 5-8
constexpr size_t kTrSamplesOff = 114;          // bytes 115-116
constexpr size_t kTrIntervalOff = 116;         // bytes 117-118

constexpr uint16_t kRevision1 = 0x0100;
constexpr uint64_t kMaxSamples = 0xFFFF;       // sample counts are 16-bit in the headers
constexpr size_t kMaxIntervalUs = 0xFFFF;      // sample interval is 16-bit microseconds

uint16_t get_be16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint32_t>(p[0]) << 8) | static_cast<uint32_t>(p[1]));
}

uint32_t get_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

void put_be16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v >> 8);
    p[1] = static_cast<uint8_t>(v & 0xFFu);
}

void put_be32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    p[2] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    p[3] = static_cast<uint8_t>(v & 0xFFu);
}

bool is_supported_format(uint16_t code) {
    return code == 1 || code == 2 || code == 3 || code == 5 || code == 8;
}

size_t bytes_per_sample(SampleFormat format) {
    switch (format) {
    case SampleFormat::IBM32:
    case SampleFormat::INT32:
    case SampleFormat::IEEE32:
        return 4;
    case SampleFormat::INT16:
        return 2;
    case SampleFormat::INT8:
        return 1;
    }
    return 4;
}

float decode_sample(const uint8_t* p, SampleFormat format) {
    switch (format) {
    case SampleFormat::IEEE32: {
        uint32_t bits = get_be32(p);
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    case SampleFormat::IBM32:
        return utils::ibm_to_ieee(get_be32(p));
    case SampleFormat::INT32:
        return static_cast<float>(static_cast<int32_t>(get_be32(p)));
    case SampleFormat::INT16:
        return static_cast<float>(static_cast<int16_t>(get_be16(p)));
    case SampleFormat::INT8:
        return static_cast<float>(static_cast<int8_t>(p[0]));
    }
    return 0.0f;
}

// Encodes a sample as big-endian IEEE32 or IBM32 (the only writable formats).
void encode_sample(uint8_t* p, float value, SampleFormat format) {
    uint32_t bits;
    if (format == SampleFormat::IBM32) {
        bits = utils::ieee_to_ibm(value);
    } else {
        std::memcpy(&bits, &value, sizeof(bits));
    }
    put_be32(p, bits);
}

// A text header is ASCII when most of its bytes are printable. EBCDIC space (0x40) is '@' in
// ASCII, so it is not counted; otherwise a blank EBCDIC header would look like ASCII.
bool looks_like_ascii(const std::array<uint8_t, kTextHeaderSize>& text) {
    size_t printable = 0;
    for (uint8_t b : text) {
        const bool printable_ascii = (b >= 0x20 && b <= 0x7E && b != 0x40);
        if (printable_ascii || b == '\t' || b == '\n' || b == '\r') {
            ++printable;
        }
    }
    return printable * 2 > text.size();
}

// ASCII to EBCDIC (code page 037) for the printable characters used in generated headers.
// Characters without a mapping become EBCDIC space.
uint8_t ascii_to_ebcdic(char c) {
    if (c >= 'A' && c <= 'I') return static_cast<uint8_t>(0xC1 + (c - 'A'));
    if (c >= 'J' && c <= 'R') return static_cast<uint8_t>(0xD1 + (c - 'J'));
    if (c >= 'S' && c <= 'Z') return static_cast<uint8_t>(0xE2 + (c - 'S'));
    if (c >= 'a' && c <= 'i') return static_cast<uint8_t>(0x81 + (c - 'a'));
    if (c >= 'j' && c <= 'r') return static_cast<uint8_t>(0x91 + (c - 'j'));
    if (c >= 's' && c <= 'z') return static_cast<uint8_t>(0xA2 + (c - 's'));
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(0xF0 + (c - '0'));
    switch (c) {
    case '.': return 0x4B;
    case '<': return 0x4C;
    case '(': return 0x4D;
    case '+': return 0x4E;
    case '|': return 0x4F;
    case '&': return 0x50;
    case '!': return 0x5A;
    case '$': return 0x5B;
    case '*': return 0x5C;
    case ')': return 0x5D;
    case ';': return 0x5E;
    case '-': return 0x60;
    case '/': return 0x61;
    case ',': return 0x6B;
    case '%': return 0x6C;
    case '_': return 0x6D;
    case '>': return 0x6E;
    case '?': return 0x6F;
    case ':': return 0x7A;
    case '#': return 0x7B;
    case '@': return 0x7C;
    case '\'': return 0x7D;
    case '=': return 0x7E;
    case '"': return 0x7F;
    default: return 0x40;
    }
}

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw std::runtime_error("SEG-Y file '" + path + "': " + what);
}

} // namespace

SegyData read_segy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open SEG-Y file '" + path + "'");
    }

    in.seekg(0, std::ios::end);
    const std::streamoff end_pos = in.tellg();
    in.seekg(0, std::ios::beg);
    if (end_pos < 0) {
        fail(path, "cannot determine file size");
    }
    const auto file_size = static_cast<uint64_t>(end_pos);
    if (file_size < kFileHeaderSize) {
        fail(path, "file is smaller than the 3600-byte textual and binary headers");
    }

    SegyData data;
    in.read(reinterpret_cast<char*>(data.raw_text_header.data()),
            static_cast<std::streamsize>(kTextHeaderSize));
    in.read(reinterpret_cast<char*>(data.raw_binary_header.data()),
            static_cast<std::streamsize>(kBinaryHeaderSize));
    if (!in) {
        fail(path, "unexpected end of file while reading headers");
    }

    const auto& bin = data.raw_binary_header;

    if (looks_like_ascii(data.raw_text_header)) {
        data.text_header.assign(reinterpret_cast<const char*>(data.raw_text_header.data()),
                                kTextHeaderSize);
    } else {
        data.text_header = parse_text_header(data.raw_text_header.data());
    }

    const uint16_t format_code = get_be16(&bin[kBinFormatOff]);
    if (!is_supported_format(format_code)) {
        fail(path, "unsupported sample format code " + std::to_string(format_code) +
                       " (supported: 1, 2, 3, 5, 8)");
    }
    data.format = static_cast<SampleFormat>(format_code);

    const auto ext_count = static_cast<int16_t>(get_be16(&bin[kBinExtCountOff]));
    if (ext_count == -1) {
        fail(path, "unsupported: variable number of extended textual headers (-1)");
    }
    if (ext_count < 0) {
        fail(path, "invalid extended textual header count " + std::to_string(ext_count));
    }
    const uint64_t data_start =
        kFileHeaderSize + static_cast<uint64_t>(ext_count) * kExtTextHeaderSize;
    if (file_size < data_start) {
        fail(path, "file is truncated: the " + std::to_string(ext_count) +
                       " extended textual header(s) extend past the end of the file");
    }
    in.seekg(static_cast<std::streamoff>(data_start), std::ios::beg);

    const size_t nsamples = get_be16(&bin[kBinSamplesOff]);
    if (nsamples == 0) {
        fail(path, "binary header declares zero samples per trace");
    }
    data.nsamples = nsamples;
    data.dt = static_cast<double>(get_be16(&bin[kBinSampleIntervalOff])) * 1e-6;

    const size_t bps = bytes_per_sample(data.format);
    const uint64_t trace_bytes = kTraceHeaderSize + static_cast<uint64_t>(nsamples) * bps;
    const uint64_t payload = file_size - data_start;
    if (payload % trace_bytes != 0) {
        fail(path, "file size " + std::to_string(file_size) + " bytes does not match " +
                       std::to_string(data_start) + " bytes of headers followed by whole traces of " +
                       std::to_string(trace_bytes) + " bytes (" + std::to_string(nsamples) +
                       " samples, format " + std::to_string(format_code) + ")");
    }

    const auto ntraces = static_cast<size_t>(payload / trace_bytes);
    data.ntraces = ntraces;
    data.raw_trace_headers.resize(ntraces);
    data.samples.resize(ntraces * nsamples);

    std::vector<uint8_t> sample_buf(nsamples * bps);
    for (size_t t = 0; t < ntraces; ++t) {
        auto& header = data.raw_trace_headers[t];
        in.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(kTraceHeaderSize));
        in.read(reinterpret_cast<char*>(sample_buf.data()), static_cast<std::streamsize>(sample_buf.size()));
        if (!in) {
            fail(path, "unexpected end of file while reading trace " + std::to_string(t));
        }

        const uint16_t trace_samples = get_be16(&header[kTrSamplesOff]);
        if (trace_samples != 0 && trace_samples != nsamples) {
            fail(path, "trace " + std::to_string(t) + " header declares " +
                           std::to_string(trace_samples) + " samples but the binary header declares " +
                           std::to_string(nsamples));
        }

        float* out = data.samples.data() + t * nsamples;
        for (size_t s = 0; s < nsamples; ++s) {
            out[s] = decode_sample(sample_buf.data() + s * bps, data.format);
        }
    }

    return data;
}

void write_segy(const std::string& path, const SegyData& data, SampleFormat out_format) {
    if (out_format != SampleFormat::IBM32 && out_format != SampleFormat::IEEE32) {
        throw std::invalid_argument("write_segy: only IBM32 and IEEE32 output formats are supported");
    }
    if (data.nsamples == 0 || data.nsamples > kMaxSamples) {
        throw std::invalid_argument("write_segy: nsamples must be in [1, 65535]");
    }
    if (data.samples.size() != data.ntraces * data.nsamples) {
        throw std::invalid_argument("write_segy: samples size does not equal ntraces * nsamples");
    }
    if (data.raw_trace_headers.size() != data.ntraces) {
        throw std::invalid_argument("write_segy: one raw trace header is required per trace");
    }
    if (get_be16(&data.raw_binary_header[kBinSamplesOff]) != data.nsamples) {
        throw std::invalid_argument(
            "write_segy: binary header samples per trace does not match nsamples");
    }
    for (size_t t = 0; t < data.ntraces; ++t) {
        const uint16_t trace_samples = get_be16(&data.raw_trace_headers[t][kTrSamplesOff]);
        if (trace_samples != 0 && trace_samples != data.nsamples) {
            throw std::invalid_argument("write_segy: trace " + std::to_string(t) +
                                        " header samples do not match nsamples");
        }
    }

    std::array<uint8_t, kBinaryHeaderSize> bin = data.raw_binary_header;
    put_be16(&bin[kBinFormatOff], static_cast<uint16_t>(out_format));
    put_be16(&bin[kBinExtCountOff], 0);  // extended textual headers are not retained

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot create SEG-Y file '" + path + "'");
    }

    out.write(reinterpret_cast<const char*>(data.raw_text_header.data()),
              static_cast<std::streamsize>(kTextHeaderSize));
    out.write(reinterpret_cast<const char*>(bin.data()), static_cast<std::streamsize>(kBinaryHeaderSize));

    const size_t bps = bytes_per_sample(out_format);
    std::vector<uint8_t> sample_buf(data.nsamples * bps);
    for (size_t t = 0; t < data.ntraces; ++t) {
        out.write(reinterpret_cast<const char*>(data.raw_trace_headers[t].data()),
                  static_cast<std::streamsize>(kTraceHeaderSize));
        const float* in = data.samples.data() + t * data.nsamples;
        for (size_t s = 0; s < data.nsamples; ++s) {
            encode_sample(sample_buf.data() + s * bps, in[s], out_format);
        }
        out.write(reinterpret_cast<const char*>(sample_buf.data()),
                  static_cast<std::streamsize>(sample_buf.size()));
    }

    out.close();
    if (!out) {
        throw std::runtime_error("failed while writing SEG-Y file '" + path + "'");
    }
}

SegyData make_segy(const std::vector<float>& samples, size_t ntraces, size_t nsamples,
                   double dt, SampleFormat format) {
    if (format != SampleFormat::IBM32 && format != SampleFormat::IEEE32) {
        throw std::invalid_argument("make_segy: only IBM32 and IEEE32 formats are supported");
    }
    if (nsamples == 0 || nsamples > kMaxSamples) {
        throw std::invalid_argument("make_segy: nsamples must be in [1, 65535]");
    }
    if (samples.size() != ntraces * nsamples) {
        throw std::invalid_argument("make_segy: samples size does not equal ntraces * nsamples");
    }
    if (!(dt > 0.0)) {
        throw std::invalid_argument("make_segy: dt must be positive");
    }
    const double interval_us = std::round(dt * 1e6);
    if (interval_us < 1.0 || interval_us > static_cast<double>(kMaxIntervalUs)) {
        throw std::invalid_argument("make_segy: dt must be between 1 us and 65535 us");
    }
    const auto dt_us = static_cast<uint16_t>(interval_us);

    SegyData data;
    data.format = format;
    data.ntraces = ntraces;
    data.nsamples = nsamples;
    data.dt = static_cast<double>(dt_us) * 1e-6;
    data.samples = samples;

    const std::string text = "C 1 CREATED BY BASALT";
    data.text_header.assign(kTextHeaderSize, ' ');
    data.text_header.replace(0, text.size(), text);
    for (size_t i = 0; i < kTextHeaderSize; ++i) {
        data.raw_text_header[i] = ascii_to_ebcdic(data.text_header[i]);
    }

    auto& bin = data.raw_binary_header;
    put_be16(&bin[kBinSampleIntervalOff], dt_us);
    put_be16(&bin[kBinSamplesOff], static_cast<uint16_t>(nsamples));
    put_be16(&bin[kBinFormatOff], static_cast<uint16_t>(format));
    put_be16(&bin[kBinRevisionOff], kRevision1);
    put_be16(&bin[kBinFixedLengthOff], 1);

    data.raw_trace_headers.resize(ntraces);
    for (size_t t = 0; t < ntraces; ++t) {
        auto& header = data.raw_trace_headers[t];
        header.fill(0);
        const auto seq = static_cast<uint32_t>(t + 1);
        put_be32(&header[kTrSeqLineOff], seq);
        put_be32(&header[kTrSeqFileOff], seq);
        put_be16(&header[kTrSamplesOff], static_cast<uint16_t>(nsamples));
        put_be16(&header[kTrIntervalOff], dt_us);
    }

    return data;
}

} // namespace basalt::io
