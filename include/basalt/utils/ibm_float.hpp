#pragma once

#include <cstdint>
#include <cmath>

namespace basalt::utils {

/// Convert a 32-bit IBM 360 float (big-endian already swapped to host order)
/// to a native IEEE 754 float.
///
/// IBM 360 format:
///   Bit  31:     Sign (0 = positive, 1 = negative)
///   Bits 30-24:  Exponent (base-16, biased by 64)
///   Bits 23-0:   Fraction (24-bit, no implicit leading bit)
///
/// IEEE 754 single:
///   Bit  31:     Sign
///   Bits 30-23:  Exponent (base-2, biased by 127)
///   Bits 22-0:   Mantissa (23-bit, implicit leading 1)
///
inline float ibm_to_ieee(uint32_t ibm) {
    // Zero check: if fraction is zero, result is 0.0 regardless of exponent
    if ((ibm & 0x00FFFFFF) == 0) {
        return 0.0f;
    }

    // Extract IBM fields
    uint32_t sign = ibm >> 31;
    int32_t  ibm_exp = static_cast<int32_t>((ibm >> 24) & 0x7F);
    uint32_t fraction = ibm & 0x00FFFFFF;

    // IBM value = (-1)^sign * fraction * 16^(ibm_exp - 64) / 2^24
    //           = (-1)^sign * fraction * 2^(4*(ibm_exp - 64) - 24)

    // Normalize: shift fraction left until the MSB (bit 23) is set.
    // Each left-shift by 1 decreases the effective binary exponent by 1.
    int32_t shift = 0;
    while ((fraction & 0x00800000) == 0) {
        fraction <<= 1;
        shift++;
    }

    // Compute IEEE binary exponent
    // IBM: value = fraction * 2^(4*(ibm_exp-64) - 24)
    // After normalization: fraction has implicit 1 at bit 23
    // IEEE exponent (unbiased) = 4*(ibm_exp - 64) - 24 + 23 - shift
    //                          = 4*ibm_exp - 256 - 1 - shift
    // IEEE biased = above + 127 = 4*ibm_exp - 130 - shift
    int32_t ieee_exp = 4 * ibm_exp - 130 - shift;

    if (ieee_exp <= 0) {
        // Underflow → denorm or zero
        return 0.0f;
    }
    if (ieee_exp >= 255) {
        // Overflow → infinity
        return sign ? -INFINITY : INFINITY;
    }

    // Strip the implicit leading 1 bit (bit 23) from the fraction
    uint32_t ieee_mantissa = fraction & 0x007FFFFF;

    // Assemble IEEE 754
    uint32_t ieee_bits = (sign << 31) 
                       | (static_cast<uint32_t>(ieee_exp) << 23) 
                       | ieee_mantissa;

    float result;
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
    __builtin_memcpy(&result, &ieee_bits, sizeof(float));
    return result;
}

/// Convert a native IEEE 754 float to 32-bit IBM 360 format.
/// Useful for round-trip testing and writing IBM-format SEG-Y files.
inline uint32_t ieee_to_ibm(float value) {
    if (value == 0.0f) {
        return 0u;
    }

    uint32_t ieee_bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
    __builtin_memcpy(&ieee_bits, &value, sizeof(uint32_t));

    uint32_t sign = ieee_bits >> 31;
    int32_t  ieee_exp = static_cast<int32_t>((ieee_bits >> 23) & 0xFF);
    uint32_t mantissa = ieee_bits & 0x007FFFFF;

    // Add implicit leading 1
    mantissa |= 0x00800000;

    // IEEE value = (-1)^sign * mantissa * 2^(ieee_exp - 127 - 23)
    // IBM:  value = (-1)^sign * fraction * 16^(ibm_exp - 64) / 2^24
    //            = (-1)^sign * fraction * 2^(4*(ibm_exp - 64) - 24)

    // So we need: 4*(ibm_exp - 64) - 24 = ieee_exp - 127 - 23
    //           : 4*ibm_exp - 256 - 24 = ieee_exp - 150
    //           : 4*ibm_exp = ieee_exp - 150 + 280 = ieee_exp + 130
    // ibm_exp = (ieee_exp + 130) / 4, but must be integer

    int32_t binary_exp = ieee_exp - 150; // unbiased binary exponent of mantissa
    // We need: value = fraction * 2^(4*ibm_exp - 280)
    // So: binary_exp = 4*ibm_exp - 280 - shift_amount
    // where shift_amount is how much we right-shift the mantissa

    // ibm_exp = ceil((binary_exp + 280) / 4) but constrained to [0, 127]
    int32_t temp = binary_exp + 280;
    int32_t ibm_exp;
    int32_t remainder;
    
    if (temp >= 0) {
        ibm_exp = (temp + 3) / 4;  // ceiling division for positive
        remainder = 4 * ibm_exp - temp;
    } else {
        ibm_exp = 0;
        remainder = -temp;
    }

    // Adjust mantissa for the remainder
    if (remainder > 0) {
        mantissa >>= remainder;
    }

    // Clamp exponent
    if (ibm_exp > 127) {
        ibm_exp = 127;
        mantissa = 0x00FFFFFF; // max fraction
    }
    if (ibm_exp < 0) {
        return 0u;
    }

    // Truncate mantissa to 24 bits
    mantissa &= 0x00FFFFFF;

    if (mantissa == 0) {
        return 0u;
    }

    return (sign << 31) 
         | (static_cast<uint32_t>(ibm_exp) << 24) 
         | mantissa;
}

} // namespace basalt::utils
