#include "basalt/utils/ibm_float.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>

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
        EXPECT_NEAR(back, v, std::fabs(v) * 1e-5f)
            << "Round-trip failed for value: " << v;
    }
}

TEST(IbmFloatTest, RoundTripNegative) {
    float values[] = {-1.0f, -42.5f, -0.25f, -1000.0f};
    for (float v : values) {
        uint32_t ibm = ieee_to_ibm(v);
        float back = ibm_to_ieee(ibm);
        EXPECT_NEAR(back, v, std::fabs(v) * 1e-5f)
            << "Round-trip failed for value: " << v;
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
