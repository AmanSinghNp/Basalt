#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/cpu_state.hpp"
#include "basalt/kernel/fft2d.hpp"
#include "basalt/kernel/fft_avx.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace basalt::kernel;

namespace {

using Clock = std::chrono::steady_clock;

// The transforms are unnormalised, so repeated in-place forward FFTs on one
// buffer overflow to inf/NaN. Every shot therefore starts from a pristine copy
// of the input. The copy is excluded from the timed region.
struct Input {
    std::vector<float> real;
    std::vector<float> imag;
};

Input make_input(size_t n, uint32_t seed) {
    Input in;
    in.real.resize(n);
    in.imag.resize(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        in.real[i] = dist(rng);
        in.imag[i] = dist(rng);
    }
    return in;
}

void restore(const Input& in, float* real, float* imag, size_t n) {
    std::memcpy(real, in.real.data(), n * sizeof(float));
    std::memcpy(imag, in.imag.data(), n * sizeof(float));
}

bool all_finite(const float* real, const float* imag, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(real[i]) || !std::isfinite(imag[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

static bool benchmark_fft(size_t n, int warmup, int iterations) {
    const Input input = make_input(n, 42);

    basalt::MemoryArena arena(n * sizeof(float) * 4);
    ComplexSoA data(arena, n);

    for (int i = 0; i < warmup; ++i) {
        restore(input, data.real, data.imag, n);
        fft_forward_avx(data.real, data.imag, n);
    }

    Clock::duration measured{};
    for (int i = 0; i < iterations; ++i) {
        restore(input, data.real, data.imag, n);
        const auto start = Clock::now();
        fft_forward_avx(data.real, data.imag, n);
        measured += Clock::now() - start;
    }

    if (!all_finite(data.real, data.imag, n)) {
        std::cerr << "FFT Size " << n << ": non-finite output\n";
        return false;
    }

    const double elapsed_sec = std::chrono::duration<double>(measured).count();
    const double avg_time_sec = elapsed_sec / iterations;
    const double throughput_mitems = (static_cast<double>(n) * iterations) / elapsed_sec / 1e6;

    std::cout << "FFT Size " << n << ": " << avg_time_sec * 1e6 << " us/op, " << throughput_mitems
              << " MItems/s" << std::endl;
    return true;
}

static bool benchmark_fft2d(size_t rows, size_t cols, int warmup, int iterations) {
    std::cout << "Initializing 2D FFT " << rows << "x" << cols << "..." << std::endl;
    const size_t total = rows * cols;
    const Input input = make_input(total, 42);

    basalt::MemoryArena arena(total * sizeof(float) * 4);
    ComplexSoA data(arena, total);
    basalt::MemoryArena scratch(total * sizeof(float) * 2 + 8192);

    for (int i = 0; i < warmup; ++i) {
        restore(input, data.real, data.imag, total);
        scratch.reset();
        fft2d_forward(data.real, data.imag, rows, cols, scratch);
    }

    Clock::duration measured{};
    for (int i = 0; i < iterations; ++i) {
        restore(input, data.real, data.imag, total);
        scratch.reset();
        const auto start = Clock::now();
        fft2d_forward(data.real, data.imag, rows, cols, scratch);
        measured += Clock::now() - start;
    }

    if (!all_finite(data.real, data.imag, total)) {
        std::cerr << "2D FFT " << rows << "x" << cols << ": non-finite output\n";
        return false;
    }

    const double elapsed_sec = std::chrono::duration<double>(measured).count();
    const double avg_ms = elapsed_sec * 1e3 / iterations;
    const double throughput_mitems = (static_cast<double>(total) * iterations) / elapsed_sec / 1e6;

    std::cout << "2D FFT " << rows << "x" << cols << ": " << avg_ms << " ms/op, "
              << throughput_mitems << " MItems/s" << std::endl;
    return true;
}

int main() {
    set_flush_to_zero();

    bool ok = true;
    std::cout << "Benchmarking AVX FFT..." << std::endl;
    const std::vector<size_t> sizes = {256, 512, 1024};
    for (size_t n : sizes) {
        const int iterations = n > 4096 ? 1000 : 10000;
        ok = benchmark_fft(n, 100, iterations) && ok;
    }

    std::cout << "Benchmarking 2D FFT..." << std::endl;
    ok = benchmark_fft2d(256, 256, 3, 50) && ok;

    if (!ok) {
        std::cerr << "fft_benchmark: non-finite results, timings are not valid\n";
        return 1;
    }
    return 0;
}
