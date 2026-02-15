#pragma once

#include <immintrin.h>

namespace basalt::kernel {

/// Set FTZ (Flush-to-Zero) and DAZ (Denormals-Are-Zero) flags in MXCSR.
/// This prevents massive performance penalties from denormal float handling.
/// MUST be called at program startup before any SIMD computation.
inline void set_flush_to_zero() {
    // FTZ bit = bit 15 (0x8000)
    // DAZ bit = bit 6  (0x0040)
    unsigned int csr = _mm_getcsr();
    csr |= 0x8040;  // FTZ | DAZ
    _mm_setcsr(csr);
}

/// Check if FTZ and DAZ flags are both set in MXCSR.
[[nodiscard]] inline bool check_ftz_daz() {
    unsigned int csr = _mm_getcsr();
    return (csr & 0x8040) == 0x8040;
}

} // namespace basalt::kernel
