#pragma once

#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/simd_mode.hpp"

namespace basalt::filter {

/// Parameters for the F-K dip filter.
struct FKParams {
    double dt = 0.004;      // Sample interval (seconds)
    double dx = 25.0;       // Trace interval (meters)
    
    // Signed apparent velocities v = f/k to mute (e.g., ground roll).
    // Weight = 1 - S((v - min)*s) * S((max - v)*s) with s = 10/taper_width, so
    // velocities inside [mute_vel_min, mute_vel_max] are muted and the band
    // edges have sigmoid tapers. The sign of v is preserved (positive v means
    // f and k have the same sign), so asymmetric bands are possible, and
    // H(f,k) == H(-f,-k) always holds. min == -max gives a symmetric |v| mute.
    // k == 0 (infinite apparent velocity, including DC) always passes.
    // Requires mute_vel_min < mute_vel_max, both finite.
    double mute_vel_min = -1500.0; 
    double mute_vel_max = 1500.0;
    
    // Width of the sigmoid taper in velocity units (m/s), finite and > 0.
    // A larger width makes the filter smoother.
    double taper_width = 100.0;
};

/// Apply F-K velocity mute to frequency-domain data.
/// 'data' is assumed to be in the f-k domain (after 2D FFT).
/// rows = number of frequencies (f)
/// cols = number of wavenumbers (k)
/// Throws std::invalid_argument if dt, dx or taper_width is not finite and > 0,
/// if the velocities are not finite with min < max, or if data is null or
/// holds fewer than rows * cols elements. rows == 0 or cols == 0 is a no-op.
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
