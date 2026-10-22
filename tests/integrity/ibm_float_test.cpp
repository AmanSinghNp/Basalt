#include "basalt/utils/ibm_float.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace basalt::utils;

// --- IBM to IEEE conversion tests ---

TEST(IbmFloatTest, ZeroConversion) {
    // All bits zero should produce 0.0
    EXPECT_FLOAT_EQ(ibm_to_ieee(0x00000000), 0.0f);

    // Zero fraction with non-zero exponent is still 0.0
    EXPECT_FLOAT_EQ(ibm_to_ieee(0x40000000), 0.0f);
}

TEST(IbmFloatTest, OnePointZero) {
    // IBM repr of 1.0:
    //   sign=0, exponent=65 (0x41), fraction=0x100000
    //   value = 0x100000 * 16^(65-64) / 2^24 = 0x100000 * 16 / 16777216
    //         = 1048576 * 16 / 16777216 = 1.0
    uint32_t ibm_one = 0x41100000;
    float result = ibm_to_ieee(ibm_one);
    EXPECT_NEAR(result, 1.0f, 1e-6f);
}

TEST(IbmFloatTest, NegativeOne) {
    // IBM -1.0: same as 1.0 but sign bit set
    uint32_t ibm_neg_one = 0xC1100000;
    float result = ibm_to_ieee(ibm_neg_one);
    EXPECT_NEAR(result, -1.0f, 1e-6f);
}

TEST(IbmFloatTest, OneHundred) {
    // IBM repr of 100.0:
    //   100 = 0x64, so fraction = 0x640000, exponent = 65 (0x41)
    //   value = 0x640000 * 16^(65-64) / 2^24 = 6553600 * 16 / 16777216 = 6.25 * 16 = 100
    // Wait: let's recalculate
    //   0x640000 = 6553600, * 16 / 16777216 = 6553600/1048576 = 6.25, * 16 = doesn't work
    //   Actually: value = fraction/2^24 * 16^(exp-64)
    //   For 100: 100 = fraction/16777216 * 16^1
    //   fraction = 100 * 16777216 / 16 = 104857600 = too big
    //   So exponent must be 66 (0x42): 100 = fraction/16777216 * 16^2 = fraction/16777216 * 256
    //   fraction = 100 * 16777216 / 256 = 6553600 = 0x640000
    uint32_t ibm_hundred = 0x42640000;
    float result = ibm_to_ieee(ibm_hundred);
    EXPECT_NEAR(result, 100.0f, 1e-4f);
}

TEST(IbmFloatTest, SmallValue) {
    // IBM repr of 0.1:
    //   0.1 = fraction/2^24 * 16^(exp-64)
    //   exp=64 (0x40): 0.1 = fraction/16777216
    //   fraction = 0.1 * 16777216 = 1677721.6 -> not exact
    //   0.1 in IBM hex: 0x40199999 (well-known)
    uint32_t ibm_point_one = 0x40199999;
    float result = ibm_to_ieee(ibm_point_one);
    EXPECT_NEAR(result, 0.1f, 1e-6f);
}

// --- Round-trip tests ---

TEST(IbmFloatTest, RoundTripPositive) {
    float values[] = {1.0f, 42.5f, 0.25f, 1000.0f, 3.14159f};
    for (float v : values) {
        uint32_t ibm = ieee_to_ibm(v);
        float back = ibm_to_ieee(ibm);
        EXPECT_NEAR(back, v, std::fabs(v) * 1e-5f) << "Round-trip failed for value: " << v;
    }
}

TEST(IbmFloatTest, RoundTripNegative) {
    float values[] = {-1.0f, -42.5f, -0.25f, -1000.0f};
    for (float v : values) {
        uint32_t ibm = ieee_to_ibm(v);
        float back = ibm_to_ieee(ibm);
        EXPECT_NEAR(back, v, std::fabs(v) * 1e-5f) << "Round-trip failed for value: " << v;
    }
}

TEST(IbmFloatTest, RoundTripZero) {
    uint32_t ibm = ieee_to_ibm(0.0f);
    EXPECT_EQ(ibm, 0u);
    EXPECT_FLOAT_EQ(ibm_to_ieee(ibm), 0.0f);
}

// --- IEEE to IBM tests ---

TEST(IbmFloatTest, IeeeToIbmKnownValues) {
    // 1.0 should map to IBM 0x41100000
    uint32_t ibm = ieee_to_ibm(1.0f);
    EXPECT_EQ(ibm, 0x41100000u);
}

// --- Edge cases: zero, subnormals, Inf/NaN, extremes ---

namespace {

float bits_to_float(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

uint32_t float_to_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

// Exact value of an IBM word as a double (all 24-bit fractions fit exactly).
double ibm_reference(uint32_t ibm) {
    const double sign = (ibm >> 31) ? -1.0 : 1.0;
    const int exp = static_cast<int>((ibm >> 24) & 0x7F);
    const double fraction = static_cast<double>(ibm & 0x00FFFFFF);
    return sign * std::ldexp(fraction, 4 * (exp - 64) - 24);
}

constexpr uint32_t kIbmMaxPos = 0x7FFFFFFFu;
constexpr uint32_t kIbmMaxNeg = 0xFFFFFFFFu;

} // namespace

TEST(IbmFloatTest, NegativeZeroIbmKeepsSign) {
    const float nz = ibm_to_ieee(0x80000000u);
    EXPECT_EQ(nz, 0.0f);
    EXPECT_TRUE(std::signbit(nz));
    const float pz = ibm_to_ieee(0x00000000u);
    EXPECT_FALSE(std::signbit(pz));
    // Zero fraction with any exponent is zero, sign preserved
    EXPECT_TRUE(std::signbit(ibm_to_ieee(0xC5000000u)));
    EXPECT_FALSE(std::signbit(ibm_to_ieee(0x45000000u)));
}

TEST(IbmFloatTest, IeeeNegativeZeroMapsToCanonicalIbmZero) {
    EXPECT_EQ(ieee_to_ibm(-0.0f), 0u);
    EXPECT_EQ(ieee_to_ibm(0.0f), 0u);
}

TEST(IbmFloatTest, IeeeSubnormalsAreNormalisedNotGivenImplicitOne) {
    // Smallest IEEE subnormal 2^-149 = 1.4e-45 used to become 5.88e-39.
    const float tiny = bits_to_float(1u);
    const uint32_t ibm = ieee_to_ibm(tiny);
    EXPECT_DOUBLE_EQ(ibm_reference(ibm), std::ldexp(1.0, -149));
    EXPECT_EQ(ibm_to_ieee(ibm), tiny);

    // All subnormal powers of two, both signs, must round-trip exactly.
    for (int e = -149; e <= -127; ++e) {
        const float v = std::ldexp(1.0f, e);
        ASSERT_EQ(ibm_to_ieee(ieee_to_ibm(v)), v) << "2^" << e;
        ASSERT_EQ(ibm_to_ieee(ieee_to_ibm(-v)), -v) << "-2^" << e;
    }
    // Largest subnormal (23 significant bits: IBM keeps 21 here, so allow 2^-20 relative error)
    // and smallest normal
    const float max_sub = bits_to_float(0x007FFFFFu);
    EXPECT_NEAR(ibm_to_ieee(ieee_to_ibm(max_sub)), max_sub, max_sub * 1e-6f);
    const float min_norm = bits_to_float(0x00800000u);
    EXPECT_EQ(ibm_to_ieee(ieee_to_ibm(min_norm)), min_norm);
}

TEST(IbmFloatTest, IbmToIeeeProducesSubnormalsInsteadOfZero) {
    // 2^-149 is IBM fraction 0x080000 (= 2^19), exponent e with
    // 2^19 * 2^(4e-280) = 2^-149 -> 4e = 112 -> e = 28 (0x1C).
    const uint32_t ibm_tiny = (28u << 24) | 0x080000u;
    EXPECT_DOUBLE_EQ(ibm_reference(ibm_tiny), std::ldexp(1.0, -149));
    const float f = ibm_to_ieee(ibm_tiny);
    EXPECT_EQ(float_to_bits(f), 1u);

    // Below half the smallest subnormal flushes to (signed) zero
    const uint32_t ibm_too_small = (20u << 24) | 0x000001u;
    EXPECT_EQ(ibm_to_ieee(ibm_too_small), 0.0f);
    EXPECT_FALSE(std::signbit(ibm_to_ieee(ibm_too_small)));
    EXPECT_TRUE(std::signbit(ibm_to_ieee(ibm_too_small | 0x80000000u)));
}

TEST(IbmFloatTest, InfAndNanSaturateToIbmMax) {
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(ieee_to_ibm(inf), kIbmMaxPos);
    EXPECT_EQ(ieee_to_ibm(-inf), kIbmMaxNeg);
    EXPECT_EQ(ieee_to_ibm(std::numeric_limits<float>::quiet_NaN()), kIbmMaxPos);
    EXPECT_EQ(ieee_to_ibm(-std::numeric_limits<float>::quiet_NaN()), kIbmMaxNeg);
    EXPECT_EQ(ieee_to_ibm(bits_to_float(0x7F800001u)), kIbmMaxPos); // signalling NaN

    // Saturated IBM max exceeds FLT_MAX, so it reads back as +-Inf
    EXPECT_EQ(ibm_to_ieee(kIbmMaxPos), inf);
    EXPECT_EQ(ibm_to_ieee(kIbmMaxNeg), -inf);
}

TEST(IbmFloatTest, ExtremeFiniteValues) {
    const float fmax = std::numeric_limits<float>::max();
    EXPECT_EQ(ibm_to_ieee(ieee_to_ibm(fmax)), fmax);
    EXPECT_EQ(ibm_to_ieee(ieee_to_ibm(-fmax)), -fmax);
    const float fmin = std::numeric_limits<float>::min();
    EXPECT_EQ(ibm_to_ieee(ieee_to_ibm(fmin)), fmin);

    // Largest finite float must not saturate: 3.4e38 << IBM max 7.2e75
    EXPECT_NE(ieee_to_ibm(fmax), kIbmMaxPos);

    // IBM values just inside / outside the float range
    // 0x60100000 is 2^124 (representable); 0x62100000 is 2^132 (overflows float)
    EXPECT_EQ(ibm_to_ieee(0x60100000u), std::ldexp(1.0f, 124));
    EXPECT_EQ(ibm_to_ieee(0x62100000u), std::numeric_limits<float>::infinity());
}

TEST(IbmFloatTest, RoundsToNearestNotTruncates) {
    // 1.0 + 2^-23 needs 24 bits: IBM exp 65 fraction 0x100000 keeps only 21
    // bits after the leading hex digit, so the low 3 bits are rounded.
    const float one_plus = bits_to_float(0x3F800001u); // 1 + 2^-23 -> rounds down to 1.0
    EXPECT_EQ(ieee_to_ibm(one_plus), 0x41100000u);
    const float three_quarters =
        bits_to_float(0x3F800004u); // 1 + 4*2^-23, exactly half unit -> ties to even
    EXPECT_EQ(ieee_to_ibm(three_quarters), 0x41100000u);
    const float above_half = bits_to_float(0x3F800005u); // just above half -> rounds up
    EXPECT_EQ(ieee_to_ibm(above_half), 0x41100001u);
    // 0.99999994 (0x3F7FFFFF) needs all 24 bits and is exact at IBM exponent 64
    EXPECT_EQ(ieee_to_ibm(bits_to_float(0x3F7FFFFFu)), 0x40FFFFFFu);
    // 2 - 2^-23 loses 3 bits and rounds up to fraction 0x200000 at exponent 65
    EXPECT_EQ(ieee_to_ibm(bits_to_float(0x3FFFFFFFu)), 0x41200000u);
}

TEST(IbmFloatTest, SweepIeeeToIbmAgainstLdexpReference) {
    uint32_t state = 0x9E3779B9u;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (int n = 0; n < 200000; ++n) {
        const uint32_t bits = next();
        const float v = bits_to_float(bits);
        if (!std::isfinite(v))
            continue;
        const uint32_t ibm = ieee_to_ibm(v);
        const double got = ibm_reference(ibm);
        const double want = static_cast<double>(v);
        if (want == 0.0) {
            ASSERT_EQ(ibm, 0u);
            continue;
        }
        // The IBM fraction must be normalised (top hex digit non-zero)
        ASSERT_NE(ibm & 0x00F00000u, 0u) << std::hex << bits;
        // Error is at most half a unit in the last IBM place (round to nearest)
        const int exp = static_cast<int>((ibm >> 24) & 0x7F);
        const double half_ulp = 0.5 * std::ldexp(1.0, 4 * (exp - 64) - 24);
        ASSERT_LE(std::fabs(got - want), half_ulp) << std::hex << bits;
        ASSERT_EQ(std::signbit(got), std::signbit(want));
    }
}

TEST(IbmFloatTest, SweepIbmToIeeeAgainstLdexpReference) {
    uint32_t state = 0x1234567u;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    const double fmax = static_cast<double>(std::numeric_limits<float>::max());
    for (int n = 0; n < 400000; ++n) {
        const uint32_t ibm = next();
        const double ref = ibm_reference(ibm);
        const float got = ibm_to_ieee(ibm);
        if (std::fabs(ref) > fmax) {
            // Beyond float range (allow the rounding boundary to land either side)
            if (std::fabs(ref) > fmax * 1.0000001) {
                ASSERT_TRUE(std::isinf(got)) << std::hex << ibm;
                ASSERT_EQ(std::signbit(got), std::signbit(ref));
            }
            continue;
        }
        // Exact for normals; correctly rounded for subnormals.
        ASSERT_EQ(got, static_cast<float>(ref)) << std::hex << ibm;
        ASSERT_EQ(std::signbit(got), std::signbit(ref)) << std::hex << ibm;
    }
}

TEST(IbmFloatTest, IbmRoundTripIsLosslessThroughIeee) {
    // IBM -> IEEE is exact for in-range normals, so IEEE -> IBM restores the word.
    uint32_t state = 0xCAFEBABEu;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    int checked = 0;
    for (int n = 0; n < 200000; ++n) {
        uint32_t ibm = next();
        ibm = (ibm & 0x80FFFFFFu) | ((40u + (ibm >> 24) % 48u) << 24); // exponent 40..87
        if ((ibm & 0x00F00000u) == 0)
            continue; // only normalised words
        const float f = ibm_to_ieee(ibm);
        ASSERT_TRUE(std::isfinite(f));
        ASSERT_EQ(ieee_to_ibm(f), ibm) << std::hex << ibm;
        ++checked;
    }
    EXPECT_GT(checked, 100000);
}
