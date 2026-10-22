#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/cpu_state.hpp"

using namespace basalt::filter;
using namespace basalt::kernel;

namespace {

using Clock = std::chrono::steady_clock;

} // namespace

static bool benchmark_fk(size_t rows, size_t cols) {
    const size_t total = rows * cols;
    basalt::MemoryArena arena(total * sizeof(float) * 2 + 4096);
    ComplexSoA data(arena, total);

    // The filter is applied in place, so every shot restores the working
    // buffer from this pristine copy. The restore is excluded from timing.
    const std::vector<float> pristine_real(total, 1.0f);
    const std::vector<float> pristine_imag(total, 0.0f);

    FKParams params;

    auto restore = [&]() {
        std::memcpy(data.real, pristine_real.data(), total * sizeof(float));
        std::memcpy(data.imag, pristine_imag.data(), total * sizeof(float));
    };

    // Warmup
    for (int i = 0; i < 10; ++i) {
        restore();
        apply_fk_filter(data, rows, cols, params);
    }

    // Measure
    int iterations = 1000;
    if (total > 1024 * 1024) iterations = 100;

    Clock::duration measured{};
    for (int i = 0; i < iterations; ++i) {
        restore();
        const auto start = Clock::now();
        apply_fk_filter(data, rows, cols, params);
        measured += Clock::now() - start;
    }

    for (size_t i = 0; i < total; ++i) {
        if (!std::isfinite(data.real[i]) || !std::isfinite(data.imag[i])) {
            std::cerr << "FK Grid " << rows << "x" << cols << ": non-finite output\n";
            return false;
        }
    }

    const double elapsed_sec = std::chrono::duration<double>(measured).count();
    const double avg_time_sec = elapsed_sec / iterations;
    const double cells = static_cast<double>(rows) * static_cast<double>(cols);
    const double throughput_mcells = (cells * iterations) / elapsed_sec / 1e6;
    const double throughput_gcells = throughput_mcells / 1000.0;

    std::cout << "FK Grid " << rows << "x" << cols << " (" << cells/1e6 << " MCells): "
              << avg_time_sec * 1000.0 << " ms/op, "
              << throughput_gcells << " GCells/s" << std::endl;
    return true;
}

int main() {
    set_flush_to_zero();

    std::cout << "Benchmarking F-K Filter Kernel..." << std::endl;
    bool ok = true;
    ok = benchmark_fk(256, 256) && ok;
    ok = benchmark_fk(512, 512) && ok;
    ok = benchmark_fk(1024, 1024) && ok;
    ok = benchmark_fk(2048, 2048) && ok;
    if (!ok) {
        std::cerr << "fk_benchmark: non-finite results, timings are not valid\n";
        return 1;
    }
    return 0;
}
