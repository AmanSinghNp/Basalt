#include "basalt/utils/trace_utils.hpp"

namespace basalt::utils {

bool is_zero_trace(const float* samples, size_t count, float epsilon) {
    if (!samples || count == 0) {
        return true;
    }

    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(samples[i]) > epsilon) {
            return false;  // Early exit on first non-zero sample
        }
    }
    return true;
}

} // namespace basalt::utils
