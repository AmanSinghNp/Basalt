#pragma once

#include "basalt/arena.hpp"
#include <cstddef>
#include <cstring>

namespace basalt::kernel {

/// Structure-of-Arrays complex number storage for SIMD-friendly access.
/// Stores real and imaginary parts in separate contiguous arrays.
/// All allocations are 64-byte aligned via MemoryArena.
struct ComplexSoA {
    float* real = nullptr;
    float* imag = nullptr;
    size_t count = 0;

    ComplexSoA() = default;

    /// Allocate from arena with 64-byte alignment
    ComplexSoA(MemoryArena& arena, size_t n) : count(n) {
        real = arena.allocate_array<float>(n);
        imag = arena.allocate_array<float>(n);
    }

    /// Fill from real-valued data (imag = 0)
    void from_real(const float* src, size_t n) {
        std::memcpy(real, src, n * sizeof(float));
        std::memset(imag, 0, n * sizeof(float));
        count = n;
    }

    /// Copy real part back to destination
    void to_real(float* dst, size_t n) const {
        std::memcpy(dst, real, n * sizeof(float));
    }

    /// Zero all data
    void clear() {
        if (real) std::memset(real, 0, count * sizeof(float));
        if (imag) std::memset(imag, 0, count * sizeof(float));
    }
};

} // namespace basalt::kernel
