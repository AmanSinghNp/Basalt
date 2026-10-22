#include "basalt/utils/trace_utils.hpp"

namespace basalt::utils {

bool is_zero_trace(const float* samples, size_t count, float epsilon) {
    if (!samples || count == 0) {
        return true;
    }

    for (size_t i = 0; i < count; ++i) {
        // Written as !(<=) so a NaN sample counts as non-zero (NaN compares
        // false with everything, which used to make all-NaN traces look blank).
        if (!(std::fabs(samples[i]) <= epsilon)) {
            return false; // Early exit on first non-zero (or NaN) sample
        }
    }
    return true;
}

} // namespace basalt::utils
