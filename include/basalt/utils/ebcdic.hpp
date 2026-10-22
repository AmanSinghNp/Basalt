#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <array>

namespace basalt::utils {

/// EBCDIC code page 037 decoding.
///
/// Output encoding decision: the std::string results are one output byte per
/// input byte (the SEG-Y textual header stays 3200 bytes long).
///   * to_ascii(): pure 7-bit output. Code points above 0x7F (accented
///     letters, C1 controls, currency signs, ...) become '?'. A NUL appears
///     only for an input byte 0x00 (the real EBCDIC NUL), never as a stand-in
///     for an unmapped printable character.
///   * to_latin1(): ISO-8859-1 bytes, lossless for all 256 inputs. Not valid
///     UTF-8 when it contains bytes >= 0x80.
class EbcdicConverter {
public:
    // Convert a raw buffer of bytes (EBCDIC cp037) to an ASCII string
    static std::string to_ascii(const void* data, size_t length);
    static void to_ascii(const void* src, char* dst, size_t length);

    // Convert to ISO-8859-1 (every cp037 byte has a Latin-1 equivalent)
    static std::string to_latin1(const void* data, size_t length);
    static void to_latin1(const void* src, char* dst, size_t length);

    /// Latin-1 code point of a single EBCDIC cp037 byte.
    static unsigned char latin1_for(uint8_t ebcdic) { return ebcdic_to_latin1_table[ebcdic]; }

private:
    static const std::array<unsigned char, 256> ebcdic_to_latin1_table;
};

/// Heuristic: does this buffer look like ASCII text rather than EBCDIC?
/// Compares how many bytes are printable ASCII (or tab/LF/CR) as-is against
/// how many are after cp037 decoding. SEG-Y textual headers are EBCDIC by
/// default but some writers emit ASCII. Returns false for null/empty input and
/// for ambiguous buffers (ties), i.e. EBCDIC is assumed unless ASCII is
/// clearly more plausible. Intended for the 3200-byte textual header.
[[nodiscard]] bool is_probably_ascii(const uint8_t* data, size_t length);

/// Decode a textual header, auto-detecting ASCII vs EBCDIC cp037 with
/// is_probably_ascii(). The result is 7-bit ASCII ('?' for anything else) and
/// has the same length as the input.
[[nodiscard]] std::string decode_text_header(const void* data, size_t length);

} // namespace basalt::utils
