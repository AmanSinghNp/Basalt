#include "basalt/utils/trace_utils.hpp"
#include "basalt/segy_format.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstring>

using namespace basalt;
using namespace basalt::utils;

// --- Dead trace detection by header ---

TEST(TraceUtilsTest, DeadTraceByHeader) {
    SegyTraceHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    
    // trace_id_code == 2 means dead trace (SEG-Y Rev 1)
    hdr.trace_id_code = 2;
    EXPECT_TRUE(is_dead_trace(hdr));
}

TEST(TraceUtilsTest, LiveTraceByHeader) {
    SegyTraceHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    
    // trace_id_code == 1 means production (live) data
    hdr.trace_id_code = 1;
    EXPECT_FALSE(is_dead_trace(hdr));
}

TEST(TraceUtilsTest, OtherTraceCodesNotDead) {
    SegyTraceHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    
    // Various other codes should not be detected as dead
    int16_t codes[] = {0, 1, 3, 4, 5, 6, 7};
    for (int16_t code : codes) {
        hdr.trace_id_code = code;
        EXPECT_FALSE(is_dead_trace(hdr)) << "trace_id_code=" << code;
    }
}

// --- Zero trace detection ---

TEST(TraceUtilsTest, AllZeroSamples) {
    std::vector<float> samples(1000, 0.0f);
    EXPECT_TRUE(is_zero_trace(samples.data(), samples.size()));
}

TEST(TraceUtilsTest, NearZeroSamples) {
    std::vector<float> samples(1000, 1e-35f);
    EXPECT_TRUE(is_zero_trace(samples.data(), samples.size(), 1e-30f));
}

TEST(TraceUtilsTest, NonZeroTrace) {
    std::vector<float> samples(1000, 0.0f);
    // Place one non-zero sample in the middle
    samples[500] = 1.0f;
    EXPECT_FALSE(is_zero_trace(samples.data(), samples.size()));
}

TEST(TraceUtilsTest, EmptyTrace) {
    EXPECT_TRUE(is_zero_trace(nullptr, 0));
}

TEST(TraceUtilsTest, SingleSampleNonZero) {
    float sample = 42.0f;
    EXPECT_FALSE(is_zero_trace(&sample, 1));
}
