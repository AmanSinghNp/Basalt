#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>

namespace basalt::utils {

class EbcdicConverter {
public:
    // Convert a raw buffer of bytes (EBCDIC) to an ASCII string
    static std::string to_ascii(const void* data, size_t length);
    static void to_ascii(const void* src, char* dst, size_t length);

private:
    static const std::array<char, 256> ebcdic_to_ascii_table;
};

} // namespace basalt::utils
