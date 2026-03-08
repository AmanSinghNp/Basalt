#pragma once

#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/simd_mode.hpp"

namespace basalt::filter {

/// Parameters for the F-K dip filter.
struct FKParams {
    double dt = 0.004;      // Sample interval (seconds)
    double dx = 25.0;       // Trace interval (meters)
    
    // Velocities to mute (e.g., ground roll < 1500 m/s)
    // Velocities inside [-vm, +vm] will be muted.
    double mute_vel_min = -1500.0; 
    double mute_vel_max = 1500.0;
    
    // Width of the sigmoid taper in velocity units (m/s)
    // A larger width makes the filter smoother.
    double taper_width = 100.0;
};

/// Apply F-K velocity mute to frequency-domain data.
/// 'data' is assumed to be in the f-k domain (after 2D FFT).
/// rows = number of frequencies (f)
/// cols = number of wavenumbers (k)
void apply_fk_filter(basalt::kernel::ComplexSoA& data, 
                     size_t rows, size_t cols, 
                     const FKParams& params);

/// Apply the F-K filter with an explicit SIMD preference.
/// Requests are best-effort and fall back to the best supported path when the
/// requested ISA is unavailable in the current build or on the current CPU.
void apply_fk_filter(basalt::kernel::ComplexSoA& data,
                     size_t rows, size_t cols,
                     const FKParams& params,
                     basalt::kernel::SimdMode simd_mode);

} // namespace basalt::filter
