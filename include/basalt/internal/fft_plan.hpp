#pragma once

// Internal FFT plumbing shared by the scalar (fft.cpp) and AVX2 (fft_avx.cpp)
// kernels and by fft2d.cpp. Not part of the installed public API.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace basalt::internal {

/// Largest transform length accepted by the public FFT entry points (2^30).
/// Keeps the bit-reversal swap table indices in 32 bits and rules out
/// size_t overflow in the helpers.
inline constexpr size_t kMaxFFTSize = size_t{1} << 30;

/// Plans for n <= kPlanCacheMaxN are cached per thread (about 12 * n bytes
/// each). Larger transforms rebuild their plan on every call.
inline constexpr size_t kPlanCacheMaxN = size_t{1} << 20;

/// Immutable per-size data: twiddles and the bit-reversal swap list.
/// Only forward-direction twiddles (cos, -sin) are stored; the inverse
/// transform uses their complex conjugates.
struct FFTPlan {
    size_t n = 0;
    size_t log2n = 0;
    /// Twiddles for butterfly stage s (block size m = 2^s, half = m/2) live at
    /// offset (half - 1), j-th entry = exp(-2*pi*i*j/m). Total length n - 1.
    std::vector<float> tw_real;
    std::vector<float> tw_imag;
    /// Flattened (i, j) pairs with i < j, bit_reverse(i) == j.
    std::vector<uint32_t> swaps;
};

/// Either a pointer into the per-thread cache or an owned, uncached plan.
class FFTPlanRef {
public:
    explicit FFTPlanRef(const FFTPlan* cached) : m_plan(cached) {}
    explicit FFTPlanRef(std::unique_ptr<const FFTPlan> owned)
        : m_plan(owned.get()), m_owned(std::move(owned)) {}

    const FFTPlan& operator*() const { return *m_plan; }
    const FFTPlan* operator->() const { return m_plan; }

private:
    const FFTPlan* m_plan;
    std::unique_ptr<const FFTPlan> m_owned;
};

/// Throws std::invalid_argument unless n is 0 or a power of two <= kMaxFFTSize,
/// and (for n > 0) real/imag are non-null and distinct. 'who' names the public
/// function in the message.
void validate_fft_args(const float* real, const float* imag, size_t n, const char* who);

/// Shared argument check for the out-of-place transposes. Returns false when
/// there is nothing to do (rows == 0 or cols == 0). Throws std::invalid_argument
/// for null pointers, rows*cols overflow, or overlapping src/dst ranges.
bool validate_transpose_args(const float* src, const float* dst,
                             size_t rows, size_t cols, const char* who);

/// Plan lookup. Precondition: n is a power of two, 2 <= n <= kMaxFFTSize.
FFTPlanRef fft_plan_for(size_t n);

/// In-place unscaled transform driven by a plan (scalar implementation).
/// 'inverse' selects conjugated twiddles. No 1/N scaling.
void fft_scalar_core(float* real, float* imag, const FFTPlan& plan, bool inverse);

/// Batched row transform: applies an unscaled FFT of length 'cols' to each of
/// 'rows' contiguous rows, then multiplies every element by 'scale' (skipped
/// when scale == 1). The plan is looked up once for the whole batch.
/// Arguments must already be validated (cols a power of two, pointers valid).
void fft_rows_avx(float* real, float* imag, size_t rows, size_t cols,
                  bool inverse, float scale);

} // namespace basalt::internal
