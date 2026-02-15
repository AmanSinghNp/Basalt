#include <gtest/gtest.h>
#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/fft2d.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace basalt::kernel;
using namespace basalt::filter;

// Helper: Linear event generator
// t = t0 + x / velocity
// Ricker wavelet source
void add_linear_event(ComplexSoA& data, size_t rows, size_t cols, 
                     double dt, double dx, double velocity, double t0, float amplitude) {
    double f_peak = 10.0; // 10 Hz Ricker to avoid spatial aliasing at 500m/s

    for (size_t j = 0; j < cols; ++j) {
 
        double offset = static_cast<double>(j) * dx;
        double arrival_time = t0 + offset / velocity;
        
        for (size_t i = 0; i < rows; ++i) {
            double time = static_cast<double>(i) * dt;
            double t = time - arrival_time;
            
            // Ricker wavelet
            // Use std::exp explicitly
            float val = amplitude * (1.0 - 2.0 * M_PI * M_PI * f_peak * f_peak * t * t) * 
                        std::exp(-M_PI * M_PI * f_peak * f_peak * t * t);
            
            size_t idx = i * cols + j;
            if (idx >= data.count) {
                // Bounds check failure
                return;
            }
            data.real[idx] += val;
        }
    }

}

// Measure total energy (sum of squares)
double total_energy(const ComplexSoA& data, size_t size) {
    double e = 0.0;
    for (size_t i = 0; i < size; ++i) {
        e += data.real[i]*data.real[i] + data.imag[i]*data.imag[i];
    }
    return e;
}

// Full Pipeline Test
class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Size
        rows = 256; // Time samples
        cols = 256; // Traces
        params.dt = 0.002; // 2ms
        params.dx = 5.0;   // 5m (finer sampling)
        
        // Let's create data arena.
        size_t bytes = rows * cols * sizeof(float) * 2 + 4096; // Some padding
        data_arena = std::make_unique<basalt::MemoryArena>(bytes * 4); // ample space
        
        data = std::make_unique<ComplexSoA>(*data_arena, rows * cols);
        
        ASSERT_NE(data->real, nullptr) << "Arena allocation failed for real array";
        ASSERT_NE(data->imag, nullptr) << "Arena allocation failed for imag array";

        // Params
        params.dt = 0.004;
        params.dx = 5.0;
        params.mute_vel_min = -1500.0;
        params.mute_vel_max = 1500.0;
        params.taper_width = 200.0;
    }

    size_t rows;
    size_t cols;
    std::unique_ptr<basalt::MemoryArena> data_arena;
    std::unique_ptr<ComplexSoA> data;
    FKParams params;
};

TEST_F(PipelineTest, RejectsGroundRoll) {
    // Clear
    std::fill_n(data->real, rows * cols, 0.0f);
    std::fill_n(data->imag, rows * cols, 0.0f);
    
    // Add Ground Roll (500 m/s)

    add_linear_event(*data, rows, cols, params.dt, params.dx, 500.0, 0.1, 1.0f);
    
    double energy_in = total_energy(*data, rows * cols);
    ASSERT_GT(energy_in, 1.0); // Ensure we added something
    
    // 1. Forward FFT
    basalt::MemoryArena scratch(rows * cols * sizeof(float) * 2 + 4096);
    fft2d_forward(data->real, data->imag, rows, cols, scratch);
    
    // 2. Filter
    apply_fk_filter(*data, rows, cols, params);
    
    // 3. Inverse FFT
    scratch.reset();
    fft2d_inverse(data->real, data->imag, rows, cols, scratch);
    
    double energy_out = total_energy(*data, rows * cols);
    
    // Expect significant reduction
    // 20dB reduction = factor of 100 in energy -> ratio < 0.01
    double ratio = energy_out / energy_in;
    EXPECT_LT(ratio, 0.01) << "Ground roll not sufficiently attenuated. Ratio: " << ratio;
}

TEST_F(PipelineTest, PreservesReflection) {
    std::fill_n(data->real, rows * cols, 0.0f);
    std::fill_n(data->imag, rows * cols, 0.0f);
    
    // Add Reflection (3000 m/s)
    add_linear_event(*data, rows, cols, params.dt, params.dx, 3000.0, 0.1, 1.0f);
    
    double energy_in = total_energy(*data, rows * cols);
    
    // Pipeline
    basalt::MemoryArena scratch(rows * cols * sizeof(float) * 2 + 4096);
    
    fft2d_forward(data->real, data->imag, rows, cols, scratch);
    
    apply_fk_filter(*data, rows, cols, params);
    
    scratch.reset();
    fft2d_inverse(data->real, data->imag, rows, cols, scratch);
    
    double energy_out = total_energy(*data, rows * cols);
    
    // Expect high preservation
    // Tapering might loose a tiny bit at edges?
    double ratio = energy_out / energy_in;
    EXPECT_GT(ratio, 0.95) << "Reflection signal lost. Ratio: " << ratio;
}

TEST_F(PipelineTest, SmallGrid) {
    // 1. Re-configure for 64x64
    rows = 64;
    cols = 64;
    params.dt = 0.004; // 4ms
    params.dx = 10.0;  // 10m
    
    // Re-allocate from existing arena (plenty of space)
    data = std::make_unique<ComplexSoA>(*data_arena, rows * cols);
    std::fill_n(data->real, rows * cols, 0.0f);
    std::fill_n(data->imag, rows * cols, 0.0f);
    
    // 2. Add Mixed Event (Reflection + Ground Roll)
    // Reflection (3000 m/s) -> Should Pass
    add_linear_event(*data, rows, cols, params.dt, params.dx, 3000.0, 0.05, 1.0f);
    
    // Ground Roll (500 m/s) -> Should Mute
    // Add to separate buffer to measure? No, just run separate passes?
    // Let's verify Ground Roll muting on small grid.
    
    // Reset and do Ground Roll
    std::fill_n(data->real, rows * cols, 0.0f);
    std::fill_n(data->imag, rows * cols, 0.0f);
    add_linear_event(*data, rows, cols, params.dt, params.dx, 500.0, 0.05, 1.0f);
    double energy_gr = total_energy(*data, rows * cols);
    
    // Pipeline
    basalt::MemoryArena scratch(rows * cols * sizeof(float) * 2 + 1024);
    fft2d_forward(data->real, data->imag, rows, cols, scratch);
    apply_fk_filter(*data, rows, cols, params);
    scratch.reset();
    fft2d_inverse(data->real, data->imag, rows, cols, scratch);
    
    double energy_out = total_energy(*data, rows * cols);
    double ratio = energy_out / energy_gr;
    
    EXPECT_LT(ratio, 0.05) << "Small Grid: Ground roll not attenuated. Ratio: " << ratio;
}
