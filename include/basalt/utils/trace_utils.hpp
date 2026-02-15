#pragma once

#include "basalt/segy_format.hpp"
#include <cstddef>
#include <cmath>

namespace basalt::utils {

/// Check if a trace is marked as dead in the SEG-Y trace header.
/// SEG-Y Rev 1: trace_id_code == 2 indicates a dead trace.
[[nodiscard]] inline bool is_dead_trace(const SegyTraceHeader& hdr) {
    return hdr.trace_id_code == 2;
}

/// Check if all samples in a trace are effectively zero.
/// This catches corrupted or blank traces that weren't flagged in the header.
/// 'epsilon' controls the threshold (default: 1e-30 for IBM float precision).
[[nodiscard]] bool is_zero_trace(const float* samples, size_t count, 
                                 float epsilon = 1e-30f);

} // namespace basalt::utils
