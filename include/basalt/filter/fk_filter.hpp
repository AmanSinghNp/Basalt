#pragma once

#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/simd_mode.hpp"

namespace basalt::filter {

/// Parameters for the F-K velocity mute.
///
/// Each f-k bin has apparent velocity v = f / k (m/s), where f is the frequency in Hz
/// (rows) and k the wavenumber in cycles per metre (cols). The velocity is signed:
/// v(-f, -k) = v(f, k), but v(f, -k) = -v(f, k), so a band such as [500, 2000] mutes
/// only one dip direction. The symmetric default [-1500, 1500] mutes both.
struct FKParams {
    double dt = 0.004;      // Sample interval (seconds)
    double dx = 25.0;       // Trace interval (meters)

    // Signed velocity band to mute, in m/s. Bins with mute_vel_min <= v <= mute_vel_max
    // are attenuated; all other bins are kept. The edges are smooth (see taper_width).
    // If mute_vel_min >= mute_vel_max, apply_fk_filter is a no-op.
    double mute_vel_min = -1500.0;
    double mute_vel_max = 1500.0;

    // Edge scale of the mute in m/s. Each band edge is a logistic sigmoid with
    // steepness s = 10 / taper_width, so a larger width gives a smoother mute.
    // taper_width <= 0 makes apply_fk_filter a no-op.
    double taper_width = 100.0;
};

/// Apply the F-K velocity mute to frequency-domain data.
///
/// 'data' is assumed to be in the f-k domain (after 2D FFT), stored row-major.
/// rows = number of frequencies (f), first axis
/// cols = number of wavenumbers (k), second axis
///
/// The per-bin weight is w = 1 - sigmoid((v - mute_vel_min) * s) * sigmoid((mute_vel_max - v) * s)
/// with s = 10 / taper_width, and each complex bin is multiplied by w.
/// Bins with k == 0 (the zero-wavenumber column, which includes DC) have v = +/-inf and are
/// always kept (w = 1), independent of the band. Exact zero is tested with a lane mask.
///
/// Nyquist bins (even sizes only: the row f = +f_N and the column k = +k_N) are their own
/// alias of both signs, so their velocity is ambiguous. Their weight is the mean of the
/// weights at v and -v. This keeps w(f, k) == w(-f, -k) for every bin and any band, so the
/// inverse transform of a real gather stays real for asymmetric bands too.
///
/// The call is a no-op (data unchanged) when taper_width <= 0, mute_vel_min >= mute_vel_max,
/// or rows == 0 or cols == 0.
///
/// The SIMD paths work in single precision. If the steepness 10 / taper_width, a band edge, or
/// a frequency or wavenumber step lies outside the normal float range, all bins are computed in
/// double precision on a scalar path instead, whatever simd_mode is requested. A bin exactly on
/// a band edge then gets the sigmoid value 1/2 for that edge, as the formula gives.
void apply_fk_filter(basalt::kernel::ComplexSoA& data,
                     size_t rows, size_t cols,
                     const FKParams& params);

/// Apply the F-K filter with an explicit SIMD preference.
/// Requests are best-effort and fall back to the best supported path when the
/// requested ISA is unavailable in the current build or on the current CPU.
/// The AVX2 and AVX-512 paths compute the same weights (agreement within 1e-6).
void apply_fk_filter(basalt::kernel::ComplexSoA& data,
                     size_t rows, size_t cols,
                     const FKParams& params,
                     basalt::kernel::SimdMode simd_mode);

} // namespace basalt::filter
