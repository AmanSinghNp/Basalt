#include "basalt/arena.hpp"
#include "basalt/kernel/fft2d.hpp"
#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/transpose_avx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Result {
    basalt::kernel::FFT2DSchedule schedule;
    size_t tile;
    double avg_ms;
};

struct Config {
    size_t size = 1024;
    int iterations = 20;
    int warmup = 3;
    std::vector<size_t> tiles;
};

bool parse_positive_int(const std::string& s, int& out) {
    try {
        size_t idx = 0;
        int v = std::stoi(s, &idx);
        if (idx != s.size() || v <= 0) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_positive_size(const std::string& s, size_t& out) {
    try {
        size_t idx = 0;
        unsigned long long v = std::stoull(s, &idx);
        if (idx != s.size() || v == 0) return false;
        out = static_cast<size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

const char* schedule_name(basalt::kernel::FFT2DSchedule schedule) {
    switch (schedule) {
    case basalt::kernel::FFT2DSchedule::Auto:
        return "auto";
    case basalt::kernel::FFT2DSchedule::LegacyColumnFirst:
        return "legacy-column-first";
    case basalt::kernel::FFT2DSchedule::FourStepRowFirst:
        return "four-step-row-first";
    }
    return "unknown";
}

std::vector<size_t> default_tiles() {
    return {64, 96, 128, 192};
}

bool append_tile(Config& cfg, const std::string& value) {
    size_t tile = 0;
    if (!parse_positive_size(value, tile)) {
        return false;
    }
    cfg.tiles.push_back(tile);
    return true;
}

double benchmark_path(const std::vector<float>& src_r,
                      const std::vector<float>& src_i,
                      std::vector<float>& work_r,
                      std::vector<float>& work_i,
                      int warmup, int iterations,
                      const std::function<void(float*, float*)>& fn) {
    for (int i = 0; i < warmup; ++i) {
        std::memcpy(work_r.data(), src_r.data(), src_r.size() * sizeof(float));
        std::memcpy(work_i.data(), src_i.data(), src_i.size() * sizeof(float));
        fn(work_r.data(), work_i.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memcpy(work_r.data(), src_r.data(), src_r.size() * sizeof(float));
        std::memcpy(work_i.data(), src_i.data(), src_i.size() * sizeof(float));
        fn(work_r.data(), work_i.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / static_cast<double>(iterations);
}

Result run_schedule_benchmark(const std::vector<float>& src_r,
                              const std::vector<float>& src_i,
                              std::vector<float>& work_r,
                              std::vector<float>& work_i,
                              size_t rows, size_t cols,
                              size_t scratch_bytes,
                              int warmup, int iterations,
                              basalt::kernel::FFT2DSchedule schedule,
                              size_t tile) {
    const auto avg_ms = benchmark_path(
        src_r, src_i, work_r, work_i, warmup, iterations,
        [&](float* r, float* im) {
            basalt::MemoryArena scratch(scratch_bytes);
            basalt::kernel::fft2d_forward(
                r, im, rows, cols, scratch,
                basalt::kernel::FFT2DConfig{schedule, tile}
            );
        }
    );

    return Result{schedule, tile, avg_ms};
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            if (!parse_positive_size(argv[++i], cfg.size)) {
                std::cerr << "Invalid --size\n";
                return 1;
            }
        } else if (arg == "--iterations" && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], cfg.iterations)) {
                std::cerr << "Invalid --iterations\n";
                return 1;
            }
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], cfg.warmup)) {
                std::cerr << "Invalid --warmup\n";
                return 1;
            }
        } else if (arg == "--tile" && i + 1 < argc) {
            if (!append_tile(cfg, argv[++i])) {
                std::cerr << "Invalid --tile\n";
                return 1;
            }
        } else if (arg == "--help") {
            std::cout << "Usage: fft2d_schedule_benchmark "
                      << "[--size N] [--iterations N] [--warmup N] [--tile N ...]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (!is_power_of_two(cfg.size)) {
        std::cerr << "--size must be a power of two\n";
        return 1;
    }
    if (cfg.tiles.empty()) {
        cfg.tiles = default_tiles();
    }

    const size_t rows = cfg.size;
    const size_t cols = cfg.size;
    const size_t total = rows * cols;

    std::vector<float> input_r(total);
    std::vector<float> input_i(total);
    std::vector<float> work_r(total);
    std::vector<float> work_i(total);
    std::vector<float> legacy_r(total);
    std::vector<float> legacy_i(total);
    std::vector<float> four_r(total);
    std::vector<float> four_i(total);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < total; ++i) {
        input_r[i] = dist(rng);
        input_i[i] = dist(rng);
    }

    const size_t scratch_bytes = total * sizeof(float) * 2 + 8192;
    std::vector<Result> results;
    results.reserve(cfg.tiles.size() * 2);
    for (size_t tile : cfg.tiles) {
        results.push_back(run_schedule_benchmark(
            input_r, input_i, work_r, work_i, rows, cols, scratch_bytes,
            cfg.warmup, cfg.iterations,
            basalt::kernel::FFT2DSchedule::LegacyColumnFirst, tile
        ));
        results.push_back(run_schedule_benchmark(
            input_r, input_i, work_r, work_i, rows, cols, scratch_bytes,
            cfg.warmup, cfg.iterations,
            basalt::kernel::FFT2DSchedule::FourStepRowFirst, tile
        ));
    }

    const Result* best_legacy = nullptr;
    const Result* best_four_step = nullptr;
    for (const Result& result : results) {
        if (result.schedule == basalt::kernel::FFT2DSchedule::LegacyColumnFirst &&
            (best_legacy == nullptr || result.avg_ms < best_legacy->avg_ms)) {
            best_legacy = &result;
        }
        if (result.schedule == basalt::kernel::FFT2DSchedule::FourStepRowFirst &&
            (best_four_step == nullptr || result.avg_ms < best_four_step->avg_ms)) {
            best_four_step = &result;
        }
    }

    // Correctness comparison on the same input.
    std::memcpy(legacy_r.data(), input_r.data(), total * sizeof(float));
    std::memcpy(legacy_i.data(), input_i.data(), total * sizeof(float));
    {
        basalt::MemoryArena scratch(scratch_bytes);
        const size_t tile = best_legacy != nullptr ? best_legacy->tile : cfg.tiles.front();
        basalt::kernel::fft2d_forward(
            legacy_r.data(), legacy_i.data(), rows, cols, scratch,
            basalt::kernel::FFT2DConfig{
                basalt::kernel::FFT2DSchedule::LegacyColumnFirst,
                tile
            }
        );
    }

    std::memcpy(four_r.data(), input_r.data(), total * sizeof(float));
    std::memcpy(four_i.data(), input_i.data(), total * sizeof(float));
    {
        basalt::MemoryArena scratch(scratch_bytes);
        const size_t tile = best_four_step != nullptr ? best_four_step->tile : cfg.tiles.front();
        basalt::kernel::fft2d_forward(
            four_r.data(), four_i.data(), rows, cols, scratch,
            basalt::kernel::FFT2DConfig{
                basalt::kernel::FFT2DSchedule::FourStepRowFirst,
                tile
            }
        );
    }

    double max_abs_real = 0.0;
    double max_abs_imag = 0.0;
    for (size_t i = 0; i < total; ++i) {
        max_abs_real = std::max(max_abs_real, std::fabs(static_cast<double>(legacy_r[i] - four_r[i])));
        max_abs_imag = std::max(max_abs_imag, std::fabs(static_cast<double>(legacy_i[i] - four_i[i])));
    }

    std::cout << "FFT2D Schedule Benchmark size=" << cfg.size
              << " iters=" << cfg.iterations
              << " warmup=" << cfg.warmup << "\n";
    std::cout << "Candidate tiles:";
    for (size_t tile : cfg.tiles) {
        std::cout << ' ' << tile;
    }
    std::cout << "\n\n";

    std::cout << std::fixed << std::setprecision(3);
    for (const Result& result : results) {
        std::cout << schedule_name(result.schedule)
                  << " tile=" << result.tile
                  << " avg=" << result.avg_ms << " ms\n";
    }
    std::cout << "\n";

    if (best_legacy != nullptr) {
        std::cout << "Best legacy: tile=" << best_legacy->tile
                  << " avg=" << best_legacy->avg_ms << " ms\n";
    }
    if (best_four_step != nullptr) {
        std::cout << "Best four-step: tile=" << best_four_step->tile
                  << " avg=" << best_four_step->avg_ms << " ms\n";
    }
    if (best_legacy != nullptr && best_four_step != nullptr && best_four_step->avg_ms > 0.0) {
        std::cout << "Best speedup (legacy/four-step): "
                  << (best_legacy->avg_ms / best_four_step->avg_ms) << "x\n";
    }
    std::cout << "Max abs diff real: " << max_abs_real << "\n";
    std::cout << "Max abs diff imag: " << max_abs_imag << "\n";

    return 0;
}
