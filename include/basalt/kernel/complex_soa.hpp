#pragma once

#include "basalt/arena.hpp"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>

namespace basalt::kernel {

/// Structure-of-Arrays complex number storage for SIMD-friendly access.
/// Stores real and imaginary parts in separate contiguous arrays.
/// All allocations are 64-byte aligned via MemoryArena.
struct ComplexSoA {
    float* real = nullptr;
    float* imag = nullptr;
    size_t count = 0;

    ComplexSoA() = default;

    /// Arena bytes needed to construct a ComplexSoA of `n` elements,
    /// including the per-allocation MemoryArena::kAliasPadding. An arena of
    /// exactly 2 * n * sizeof(float) bytes is NOT enough. Returns SIZE_MAX if
    /// not representable (the constructor then throws std::bad_alloc).
    [[nodiscard]] static size_t required_arena_bytes(size_t n) noexcept {
        if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
            return std::numeric_limits<size_t>::max();
        }
        return MemoryArena::required_bytes({n * sizeof(float), n * sizeof(float)});
    }

    /// Allocate from arena with 64-byte alignment.
    /// Throws std::bad_alloc if the arena cannot hold both arrays (see
    /// required_arena_bytes()); never leaves a null pointer behind. On failure
    /// the arena may keep the space taken by the first array until reset().
    ComplexSoA(MemoryArena& arena, size_t n) {
        float* r = arena.allocate_array<float>(n);
        if (!r) throw std::bad_alloc();
        float* im = arena.allocate_array<float>(n);
        if (!im) throw std::bad_alloc();
        real = r;
        imag = im;
        count = n;
    }

    /// Fill from real-valued data (imag = 0)
    void from_real(const float* src, size_t n) {
        assert(n <= count && "from_real: n exceeds allocated count");
        std::memcpy(real, src, n * sizeof(float));
        std::memset(imag, 0, n * sizeof(float));
    }

    /// Copy real part back to destination
    void to_real(float* dst, size_t n) const {
        assert(n <= count && "to_real: n exceeds allocated count");
        std::memcpy(dst, real, n * sizeof(float));
    }

    /// Zero all data
    void clear() {
        if (real) std::memset(real, 0, count * sizeof(float));
        if (imag) std::memset(imag, 0, count * sizeof(float));
    }
};

} // namespace basalt::kernel
