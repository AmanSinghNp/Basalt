#pragma once

#include "basalt/segy_format.hpp"
#include <string>
#include <vector>

namespace basalt::io {

// Reads the 3200-byte EBCDIC header and converts it to an ASCII string.
// 'buffer' must be at least 3200 bytes.
std::string parse_text_header(const void* buffer);

// Parses the 400-byte Binary header.
// Handles Big-Endian to Host-Endian conversion.
SegyBinaryHeader parse_binary_header(const void* buffer);

// Parses the 240-byte Trace header.
// Handles Big-Endian to Host-Endian conversion.
SegyTraceHeader parse_trace_header(const void* buffer);

} // namespace basalt::io
