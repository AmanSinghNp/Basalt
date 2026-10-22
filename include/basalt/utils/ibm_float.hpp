#pragma once

#include <cstdint>
#include <cstring>
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
/// Special cases: a zero fraction is zero for any exponent and keeps its sign
/// (0x80000000 -> -0.0f). Magnitudes below the IEEE normal range become IEEE
/// subnormals (rounded to nearest even) and flush to zero only when below half
/// the smallest subnormal. Magnitudes above FLT_MAX (IBM reaches ~7.2e75)
/// become +-Inf.
///
/// IEEE 754 single:
///   Bit  31:     Sign
///   Bits 30-23:  Exponent (base-2, biased by 127)
///   Bits 22-0:   Mantissa (23-bit, implicit leading 1)
///
inline float ibm_to_ieee(uint32_t ibm) {
    const uint32_t sign = ibm >> 31;
    const int32_t  ibm_exp = static_cast<int32_t>((ibm >> 24) & 0x7F);
    const uint32_t fraction = ibm & 0x00FFFFFF;

    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
    uint32_t ieee_bits = sign << 31;

    if (fraction == 0) {
        // Zero regardless of exponent; keep the sign bit (-0.0f).
        float zero;
        std::memcpy(&zero, &ieee_bits, sizeof(float));
        return zero;
    }

    // IBM value = (-1)^sign * fraction * 16^(ibm_exp - 64) / 2^24
    //           = (-1)^sign * fraction * 2^(4*ibm_exp - 280)

    // Normalize: shift fraction left until the MSB (bit 23) is set.
    // Each left-shift by 1 decreases the effective binary exponent by 1.
    uint32_t norm = fraction;
    int32_t shift = 0;
    while ((norm & 0x00800000) == 0) {
        norm <<= 1;
        shift++;
    }

    // IEEE biased exponent of the normalized value:
    // unbiased = 4*ibm_exp - 280 + 23 - shift, biased = unbiased + 127
    const int32_t ieee_exp = 4 * ibm_exp - 130 - shift;

    if (ieee_exp >= 255) {
        // Overflow -> infinity
        ieee_bits |= 0x7F800000u;
    } else if (ieee_exp >= 1) {
        // Normal: strip the implicit leading 1 bit (bit 23). The 24-bit
        // fraction fits the IEEE significand exactly, no rounding needed.
        ieee_bits |= (static_cast<uint32_t>(ieee_exp) << 23) | (norm & 0x007FFFFF);
    } else {
        // Subnormal: value = fraction * 2^(4*ibm_exp - 280) and the IEEE
        // subnormal unit is 2^-149, so mantissa = fraction >> (131 - 4*ibm_exp),
        // rounded to nearest even. A carry into bit 23 correctly yields the
        // smallest normal number.
        // (r can be <= 0 for unnormalised fractions, which is then an exact
        // left shift.)
        const int32_t r = 131 - 4 * ibm_exp;
        uint32_t mant = 0;
        if (r <= 0) {
            mant = fraction << (-r);
        } else if (r <= 25) {
            const uint64_t half = uint64_t{1} << (r - 1);
            const uint64_t rem = fraction & ((uint64_t{1} << r) - 1);
            mant = static_cast<uint32_t>(fraction >> r);
            if (rem > half || (rem == half && (mant & 1u))) {
                mant++;
            }
        }
        ieee_bits |= mant;
    }

    float result;
    std::memcpy(&result, &ieee_bits, sizeof(float));
    return result;
}

/// Convert a native IEEE 754 float to 32-bit IBM 360 format.
/// Useful for round-trip testing and writing IBM-format SEG-Y files.
///
/// Behaviour:
///   * +-0.0f -> 0x00000000 (canonical IBM true zero; the sign of -0 is lost).
///   * IEEE subnormals are normalised (IBM range is far wider than IEEE, so
///     every finite float is representable without under/overflow).
///   * The 24-bit IEEE significand is rounded to nearest even into the IBM
///     fraction, which carries 21 to 24 significant bits depending on the
///     leading hex digit.
///   * +-Inf and NaN cannot be represented: they saturate to the largest IBM
///     magnitude, 0x7FFFFFFF / 0xFFFFFFFF (about 7.2e75), with the sign bit of
///     the input preserved (NaN keeps its own sign bit).
inline uint32_t ieee_to_ibm(float value) {
    uint32_t ieee_bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
    std::memcpy(&ieee_bits, &value, sizeof(uint32_t));

    const uint32_t sign = ieee_bits >> 31;
    const int32_t  ieee_exp = static_cast<int32_t>((ieee_bits >> 23) & 0xFF);
    uint32_t mantissa = ieee_bits & 0x007FFFFF;

    if (ieee_exp == 0xFF) {
        // Inf or NaN: saturate.
        return (sign << 31) | 0x7FFFFFFFu;
    }

    // value = mantissa * 2^binary_exp with mantissa in [2^23, 2^24)
    int32_t binary_exp;
    if (ieee_exp == 0) {
        if (mantissa == 0) {
            return 0u;  // +-0
        }
        // Subnormal: no implicit 1; normalise.
        binary_exp = -149;
        while ((mantissa & 0x00800000) == 0) {
            mantissa <<= 1;
            binary_exp--;
        }
    } else {
        mantissa |= 0x00800000;  // implicit leading 1
        binary_exp = ieee_exp - 150;
    }

    // IBM: value = fraction * 2^(4*ibm_exp - 280). Pick the smallest ibm_exp
    // such that the fraction needs a right shift r in [0, 3] only.
    // For all finite floats ibm_exp lies in [27, 96], well inside [0, 127].
    const int32_t temp = binary_exp + 280;          // always > 0 here
    int32_t ibm_exp = (temp + 3) / 4;                // ceiling division
    const int32_t r = 4 * ibm_exp - temp;            // 0..3

    uint32_t fraction = mantissa >> r;
    if (r > 0) {
        const uint32_t rem = mantissa & ((1u << r) - 1u);
        const uint32_t half = 1u << (r - 1);
        if (rem > half || (rem == half && (fraction & 1u))) {
            // Cannot carry out of 24 bits: after a right shift by r >= 1 the
            // fraction is at most 2^(24 - r) - 1.
            fraction++;
        }
    }

    return (sign << 31)
         | (static_cast<uint32_t>(ibm_exp) << 24)
         | fraction;
}

} // namespace basalt::utils
