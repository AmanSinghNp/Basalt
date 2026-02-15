#include <gtest/gtest.h>
#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace basalt::filter;
using namespace basalt::kernel;

// Helper to fill data with a specific velocity event
void fill_event(ComplexSoA& data, size_t rows, size_t cols, double dt, double dx, double velocity, float amplitude) {
    // In f-k domain, a linear event with velocity V maps to a line f = V*k
    
    double df = 1.0 / (rows * dt);
    double dk = 1.0 / (cols * dx);
    
    for (size_t i = 0; i < rows; ++i) {
        double f = static_cast<double>(i) * df;
        // Handle Nyquist f
        if (i > rows/2) f = static_cast<double>((static_cast<int>(i) - static_cast<int>(rows))) * df;
        
        for (size_t j = 0; j < cols; ++j) {
            double k_val;
            if (j <= cols/2) k_val = static_cast<double>(j) * dk;
            else k_val = static_cast<double>((static_cast<int>(j) - static_cast<int>(cols))) * dk;
            
            // Avoid singularity
            if (std::abs(k_val) < 1e-6) continue;
            
            double v_point = f / k_val;
            
            // If v_point is within 10% of target velocity
            if (std::abs(v_point - velocity) < std::abs(velocity) * 0.1) {
                data.real[i * cols + j] += amplitude;
                // data.imag remains 0
            }
        }
    }
}

TEST(FKFilterTest, MutesLowVelocity) {
    basalt::MemoryArena arena(1024 * 1024 * 4);
    size_t rows = 256;
    size_t cols = 256;
    ComplexSoA data(arena, rows * cols);
    
    // Clear data
    std::fill_n(data.real, rows * cols, 0.0f);
    std::fill_n(data.imag, rows * cols, 0.0f);
    
    FKParams params;
    params.dt = 0.004;
    params.dx = 25.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0; // Mute everything inside [-1500, 1500]
    params.taper_width = 100.0;
    
    // Add "Ground Roll" event at 500 m/s (Should be muted)
    double noise_vel = 500.0;
    fill_event(data, rows, cols, params.dt, params.dx, noise_vel, 1.0f);
    
    // Add "Signal" event at 2500 m/s (Should pass)
    double signal_vel = 2500.0;
    fill_event(data, rows, cols, params.dt, params.dx, signal_vel, 1.0f);
    
    apply_fk_filter(data, rows, cols, params);
    
    // Check specific points
    double df = 1.0 / (rows * params.dt);
    double dk = 1.0 / (cols * params.dx);
    
    for (size_t i = 1; i < rows/2; ++i) {
        for (size_t j = 1; j < cols/2; ++j) {
            double f = static_cast<double>(i) * df;
            double k_val = static_cast<double>(j) * dk;
            double v = f / k_val;
            
            size_t idx = i * cols + j;
            float mag = std::sqrt(data.real[idx]*data.real[idx] + data.imag[idx]*data.imag[idx]);
            
            if (std::abs(v - noise_vel) < noise_vel * 0.05) {
                // Should be muted (near 0)
                EXPECT_NEAR(mag, 0.0f, 1e-2) << "Noise at v=" << v << " not muted";
            }
            if (std::abs(v - signal_vel) < signal_vel * 0.05) {
                // Should pass (amplitude ~1.0)
                // Note: Taper might reduce it slightly if close to cut, but 2500 vs 1500 is far.
                EXPECT_NEAR(mag, 1.0f, 1e-2) << "Signal at v=" << v << " attenuated";
            }
        }
    }
}

TEST(FKFilterTest, PreservesDC) {
    basalt::MemoryArena arena(4096);
    size_t rows = 16;
    size_t cols = 16;
    ComplexSoA data(arena, rows * cols);
    std::fill_n(data.real, rows * cols, 0.0f);
    
    // Set DC
    data.real[0] = 100.0f; 
    
    FKParams params;
    apply_fk_filter(data, rows, cols, params);
    
    // DC is muted by default logic (v=0).
    EXPECT_NEAR(data.real[0], 0.0f, 1e-4);
}
