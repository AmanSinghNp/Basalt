#include "basalt/kernel/fft.hpp"
#include "basalt/internal/fft_plan.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace basalt::internal {

namespace {

/// Compute log2(n) for a power of two n (size_t arithmetic, no UB for big n).
size_t log2_pow2(size_t n) {
    size_t result = 0;
    while ((size_t{1} << result) < n) {
        ++result;
    }
    return result;
}

std::unique_ptr<FFTPlan> build_plan(size_t n) {
    auto plan = std::make_unique<FFTPlan>();
    plan->n = n;
    plan->log2n = log2_pow2(n);

    // Twiddles of the last stage (m = n): exp(-2*pi*i*j/n), j < n/2.
    // Every earlier stage (m = n / 2^k) is a strided subset of this table.
    const size_t top = n / 2;
    std::vector<float> top_re(top);
    std::vector<float> top_im(top);
    const double step = -2.0 * M_PI / static_cast<double>(n);
    for (size_t j = 0; j < top; ++j) {
        const double angle = step * static_cast<double>(j);
        top_re[j] = static_cast<float>(std::cos(angle));
        top_im[j] = static_cast<float>(std::sin(angle));
    }

    plan->tw_real.resize(n - 1);
    plan->tw_imag.resize(n - 1);
    for (size_t half = 1; half <= top; half <<= 1) {
        const size_t stride = top / half;
        float* dst_re = plan->tw_real.data() + (half - 1);
        float* dst_im = plan->tw_imag.data() + (half - 1);
        for (size_t j = 0; j < half; ++j) {
            dst_re[j] = top_re[j * stride];
            dst_im[j] = top_im[j * stride];
        }
    }

    // Bit-reversal permutation as a list of swap pairs (i < rev(i)).
    std::vector<uint32_t> rev(n);
    rev[0] = 0;
    const size_t top_bit = plan->log2n - 1;
    for (size_t i = 1; i < n; ++i) {
        rev[i] = static_cast<uint32_t>((rev[i >> 1] >> 1) | ((i & 1) << top_bit));
    }
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i < rev[i])
            ++count;
    }
    plan->swaps.reserve(count * 2);
    for (size_t i = 0; i < n; ++i) {
        if (i < rev[i]) {
            plan->swaps.push_back(static_cast<uint32_t>(i));
            plan->swaps.push_back(rev[i]);
        }
    }
    return plan;
}

constexpr size_t kMaxCachedLog2 = 20;
static_assert(kPlanCacheMaxN == (size_t{1} << kMaxCachedLog2), "cache cap mismatch");

} // anonymous namespace

void validate_fft_args(const float* real, const float* imag, size_t n, const char* who) {
    if (n == 0) {
        return; // Empty transform: documented no-op, pointers are not inspected.
    }
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument(std::string(who) + ": n=" + std::to_string(n) +
                                    " is not a power of two");
    }
    if (n > kMaxFFTSize) {
        throw std::invalid_argument(std::string(who) + ": n=" + std::to_string(n) +
                                    " exceeds the maximum supported size 2^30");
    }
    if (real == nullptr || imag == nullptr) {
        throw std::invalid_argument(std::string(who) + ": null data pointer");
    }
    if (real == imag) {
        throw std::invalid_argument(std::string(who) + ": real and imag must not alias");
    }
}

FFTPlanRef fft_plan_for(size_t n) {
    if (n > kPlanCacheMaxN) {
        return FFTPlanRef(std::unique_ptr<const FFTPlan>(build_plan(n)));
    }
    // One slot per power of two: bounded (<= 21 plans per thread, no hashing).
    thread_local std::array<std::unique_ptr<const FFTPlan>, kMaxCachedLog2 + 1> cache;
    const size_t slot = log2_pow2(n);
    if (!cache[slot]) {
        cache[slot] = build_plan(n);
    }
    return FFTPlanRef(cache[slot].get());
}

void fft_scalar_core(float* real, float* imag, const FFTPlan& plan, bool inverse) {
    const size_t n = plan.n;

    // --- Step 1: Bit-reversal permutation (precomputed swap list) ---
    const uint32_t* sw = plan.swaps.data();
    const size_t num_swaps = plan.swaps.size() / 2;
    for (size_t p = 0; p < num_swaps; ++p) {
        const size_t i = sw[2 * p];
        const size_t j = sw[2 * p + 1];
        std::swap(real[i], real[j]);
        std::swap(imag[i], imag[j]);
    }

    // --- Step 2: Butterfly stages (block size doubles: 2, 4, ... n) ---
    // Forward uses W = e^(-2*pi*i*j/m); inverse uses its conjugate.
    const float sign = inverse ? -1.0f : 1.0f;
    for (size_t s = 1; s <= plan.log2n; ++s) {
        const size_t m = size_t{1} << s;
        const size_t half_m = m >> 1;
        const float* tw_re = plan.tw_real.data() + (half_m - 1);
        const float* tw_im = plan.tw_imag.data() + (half_m - 1);

        for (size_t k = 0; k < n; k += m) {
            for (size_t j = 0; j < half_m; ++j) {
                const float wr = tw_re[j];
                const float wi = sign * tw_im[j];

                const size_t even = k + j;
                const size_t odd = even + half_m;

                // t = W * X[odd]
                const float tr = wr * real[odd] - wi * imag[odd];
                const float ti = wr * imag[odd] + wi * real[odd];

                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] = real[even] + tr;
                imag[even] = imag[even] + ti;
            }
        }
    }
}

} // namespace basalt::internal

namespace basalt::kernel {

namespace {

void fft_impl(float* real, float* imag, size_t n, bool inverse, const char* who) {
    internal::validate_fft_args(real, imag, n, who);
    if (n < 2) {
        return; // n == 0: empty; n == 1: identity (1/N scaling is 1)
    }

    const internal::FFTPlanRef plan = internal::fft_plan_for(n);
    internal::fft_scalar_core(real, imag, *plan, inverse);

    // --- Step 3: Scale by 1/N for inverse ---
    if (inverse) {
        const float scale = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }
}

} // anonymous namespace

void fft_forward(float* real, float* imag, size_t n) {
    fft_impl(real, imag, n, false, "fft_forward");
}

void fft_inverse(float* real, float* imag, size_t n) {
    fft_impl(real, imag, n, true, "fft_inverse");
}

} // namespace basalt::kernel
