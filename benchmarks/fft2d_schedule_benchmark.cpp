#include "basalt/arena.hpp"
#include "basalt/kernel/fft2d.hpp"
#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/transpose_avx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using basalt::kernel::FFT2DSchedule;

struct Candidate {
    FFT2DSchedule schedule;
    size_t tile;
    double min_ms = std::numeric_limits<double>::infinity();
    double total_ms = 0.0;
    long shots = 0;
    double avg_ms() const { return shots > 0 ? total_ms / static_cast<double>(shots) : 0.0; }
};

struct Config {
    size_t rows = 1024;
    size_t cols = 1024;
    int iterations = 20;
    int warmup = 3;
    int rounds = 3;
    bool inverse = false;
    std::vector<size_t> tiles;
};

bool parse_positive_int(const std::string& s, int& out) {
    try {
        size_t idx = 0;
        int v = std::stoi(s, &idx);
        if (idx != s.size() || v <= 0)
            return false;
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
        if (idx != s.size() || v == 0)
            return false;
        out = static_cast<size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

const char* schedule_name(FFT2DSchedule schedule) {
    switch (schedule) {
    case FFT2DSchedule::Auto:
        return "auto";
    case FFT2DSchedule::LegacyColumnFirst:
        return "legacy-column-first";
    case FFT2DSchedule::FourStepRowFirst:
        return "four-step-row-first";
    case FFT2DSchedule::RowFirstColumnStrips:
        return "row-first-column-strips";
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

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            if (!parse_positive_size(argv[++i], cfg.rows)) {
                std::cerr << "Invalid --size\n";
                return 1;
            }
            cfg.cols = cfg.rows;
        } else if (arg == "--rows" && i + 1 < argc) {
            if (!parse_positive_size(argv[++i], cfg.rows)) {
                std::cerr << "Invalid --rows\n";
                return 1;
            }
        } else if (arg == "--cols" && i + 1 < argc) {
            if (!parse_positive_size(argv[++i], cfg.cols)) {
                std::cerr << "Invalid --cols\n";
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
        } else if (arg == "--rounds" && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], cfg.rounds)) {
                std::cerr << "Invalid --rounds\n";
                return 1;
            }
        } else if (arg == "--tile" && i + 1 < argc) {
            if (!append_tile(cfg, argv[++i])) {
                std::cerr << "Invalid --tile\n";
                return 1;
            }
        } else if (arg == "--inverse") {
            cfg.inverse = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: fft2d_schedule_benchmark "
                << "[--size N | --rows N --cols N] [--inverse] [--iterations N] [--warmup N] "
                << "[--rounds N] [--tile N ...]\n"
                << "Candidates are timed interleaved over --rounds rounds of --iterations shots;\n"
                << "min and mean per shot are reported.\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (!is_power_of_two(cfg.rows) || !is_power_of_two(cfg.cols)) {
        std::cerr << "--size / --rows / --cols must be powers of two\n";
        return 1;
    }
    if (cfg.tiles.empty()) {
        cfg.tiles = default_tiles();
    }

    const size_t rows = cfg.rows;
    const size_t cols = cfg.cols;
    const size_t total = rows * cols;

    std::vector<float> input_r(total), input_i(total);
    std::vector<float> work_r(total), work_i(total);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < total; ++i) {
        input_r[i] = dist(rng);
        input_i[i] = dist(rng);
    }

    std::vector<Candidate> candidates;
    for (size_t tile : cfg.tiles) {
        candidates.push_back({FFT2DSchedule::LegacyColumnFirst, tile});
        candidates.push_back({FFT2DSchedule::FourStepRowFirst, tile});
    }
    candidates.push_back({FFT2DSchedule::RowFirstColumnStrips, 0}); // tile is ignored
    candidates.push_back({FFT2DSchedule::Auto, 0});

    basalt::MemoryArena scratch(basalt::kernel::fft2d_scratch_bytes(rows, cols));
    auto run_once = [&](const Candidate& c) {
        scratch.reset();
        const basalt::kernel::FFT2DConfig fc{c.schedule, c.tile == 0 ? size_t{128} : c.tile};
        if (cfg.inverse) {
            basalt::kernel::fft2d_inverse(work_r.data(), work_i.data(), rows, cols, scratch, fc);
        } else {
            basalt::kernel::fft2d_forward(work_r.data(), work_i.data(), rows, cols, scratch, fc);
        }
    };
    auto restore = [&]() {
        std::memcpy(work_r.data(), input_r.data(), total * sizeof(float));
        std::memcpy(work_i.data(), input_i.data(), total * sizeof(float));
    };

    for (Candidate& c : candidates) {
        for (int i = 0; i < cfg.warmup; ++i) {
            restore();
            run_once(c);
        }
    }

    // Interleave: every round visits every candidate, so slow phases of a noisy
    // shared machine hit all candidates alike. The restore is not timed.
    for (int round = 0; round < cfg.rounds; ++round) {
        for (Candidate& c : candidates) {
            for (int i = 0; i < cfg.iterations; ++i) {
                restore();
                const auto start = std::chrono::steady_clock::now();
                run_once(c);
                const std::chrono::duration<double, std::milli> dt =
                    std::chrono::steady_clock::now() - start;
                c.min_ms = std::min(c.min_ms, dt.count());
                c.total_ms += dt.count();
                ++c.shots;
            }
        }
    }

    // Correctness: compare every candidate with the first one on the same input.
    restore();
    run_once(candidates.front());
    const std::vector<float> ref_r = work_r;
    const std::vector<float> ref_i = work_i;
    double worst_diff = 0.0;
    double ref_max = 0.0;
    for (size_t i = 0; i < total; ++i) {
        ref_max = std::max({ref_max, std::fabs(static_cast<double>(ref_r[i])),
                            std::fabs(static_cast<double>(ref_i[i]))});
    }
    for (const Candidate& c : candidates) {
        restore();
        run_once(c);
        for (size_t i = 0; i < total; ++i) {
            worst_diff = std::max({worst_diff, std::fabs(static_cast<double>(work_r[i]) - ref_r[i]),
                                   std::fabs(static_cast<double>(work_i[i]) - ref_i[i])});
        }
    }

    std::cout << "FFT2D Schedule Benchmark " << rows << "x" << cols
              << (cfg.inverse ? " inverse" : " forward") << " iters=" << cfg.iterations
              << " rounds=" << cfg.rounds << " warmup=" << cfg.warmup << "\n\n";
    std::cout << std::fixed << std::setprecision(3);
    const Candidate* best_legacy = nullptr;
    const Candidate* best_four_step = nullptr;
    const Candidate* strips = nullptr;
    for (const Candidate& c : candidates) {
        std::cout << std::left << std::setw(26) << schedule_name(c.schedule) << std::right
                  << " tile=" << std::setw(4) << c.tile << "  min=" << std::setw(9) << c.min_ms
                  << " ms"
                  << "  avg=" << std::setw(9) << c.avg_ms() << " ms\n";
        if (c.schedule == FFT2DSchedule::LegacyColumnFirst &&
            (best_legacy == nullptr || c.min_ms < best_legacy->min_ms)) {
            best_legacy = &c;
        }
        if (c.schedule == FFT2DSchedule::FourStepRowFirst &&
            (best_four_step == nullptr || c.min_ms < best_four_step->min_ms)) {
            best_four_step = &c;
        }
        if (c.schedule == FFT2DSchedule::RowFirstColumnStrips) {
            strips = &c;
        }
    }
    std::cout << "\n";
    if (best_legacy != nullptr) {
        std::cout << "Best legacy:     tile=" << best_legacy->tile << " min=" << best_legacy->min_ms
                  << " ms\n";
    }
    if (best_four_step != nullptr) {
        std::cout << "Best four-step:  tile=" << best_four_step->tile
                  << " min=" << best_four_step->min_ms << " ms\n";
    }
    if (strips != nullptr) {
        std::cout << "Column strips:   min=" << strips->min_ms << " ms\n";
        const double best_old =
            std::min(best_legacy != nullptr ? best_legacy->min_ms : 1e300,
                     best_four_step != nullptr ? best_four_step->min_ms : 1e300);
        if (strips->min_ms > 0.0) {
            std::cout << "Speedup of strips over best transpose schedule: "
                      << (best_old / strips->min_ms) << "x\n";
        }
    }
    std::cout << "Max abs diff vs first candidate: " << std::scientific << worst_diff
              << " (max |value| " << ref_max << ")\n";

    return 0;
}
