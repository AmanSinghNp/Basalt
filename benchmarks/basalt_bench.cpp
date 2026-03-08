#include "basalt/arena.hpp"
#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/fft2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct Config {
    size_t size = 1024;
    size_t shots = 100;
    size_t warmup = 5;
    uint32_t seed = 1337;
    std::string json_out;
};

struct Stats {
    double total_seconds = 0.0;
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double throughput_gb_s = 0.0;
    double bytes_per_shot = 0.0;
    uint64_t checksum_real = 0;
    uint64_t checksum_imag = 0;
};

bool is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

bool parse_u64(const std::string& value, size_t& out) {
    try {
        size_t idx = 0;
        unsigned long long parsed = std::stoull(value, &idx);
        if (idx != value.size()) return false;
        out = static_cast<size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_u32(const std::string& value, uint32_t& out) {
    try {
        size_t idx = 0;
        unsigned long parsed = std::stoul(value, &idx);
        if (idx != value.size()) return false;
        out = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

void print_usage() {
    std::cout
        << "Usage: basalt_bench [options]\n"
        << "Options:\n"
        << "  --size <N>        Grid size (NxN, power of two). Default: 1024\n"
        << "  --shots <N>       Timed shots. Default: 100\n"
        << "  --warmup <N>      Warmup shots. Default: 5\n"
        << "  --seed <N>        RNG seed. Default: 1337\n"
        << "  --json-out <path> Write machine-readable JSON summary.\n"
        << "  --help            Show this help.\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            ++i;
            return argv[i];
        };

        if (arg == "--help") {
            print_usage();
            return false;
        }
        if (arg == "--size") {
            const char* v = require_value("--size");
            if (!v || !parse_u64(v, cfg.size)) return false;
            continue;
        }
        if (arg == "--shots") {
            const char* v = require_value("--shots");
            if (!v || !parse_u64(v, cfg.shots)) return false;
            continue;
        }
        if (arg == "--warmup") {
            const char* v = require_value("--warmup");
            if (!v || !parse_u64(v, cfg.warmup)) return false;
            continue;
        }
        if (arg == "--seed") {
            const char* v = require_value("--seed");
            if (!v || !parse_u32(v, cfg.seed)) return false;
            continue;
        }
        if (arg == "--json-out") {
            const char* v = require_value("--json-out");
            if (!v) return false;
            cfg.json_out = v;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        return false;
    }
    return true;
}

double percentile_ms(const std::vector<double>& latencies, double p) {
    if (latencies.empty()) return 0.0;
    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    if (sorted.size() == 1) return sorted.front();

    double pos = p * static_cast<double>(sorted.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    double weight = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - weight) + sorted[hi] * weight;
}

uint64_t fnv1a_hash_floats(const float* data, size_t count, uint64_t seed = 1469598103934665603ULL) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    size_t total = count * sizeof(float);
    uint64_t h = seed;
    for (size_t i = 0; i < total; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string to_hex(uint64_t value) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return oss.str();
}

void write_json(std::ostream& out, const Config& cfg, const Stats& s) {
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"size\": " << cfg.size << ",\n";
    out << "    \"shots\": " << cfg.shots << ",\n";
    out << "    \"warmup\": " << cfg.warmup << ",\n";
    out << "    \"seed\": " << cfg.seed << ",\n";
    out << "    \"bytes_per_shot\": " << s.bytes_per_shot << "\n";
    out << "  },\n";
    out << "  \"summary\": {\n";
    out << "    \"total_seconds\": " << s.total_seconds << ",\n";
    out << "    \"mean_ms\": " << s.mean_ms << ",\n";
    out << "    \"stddev_ms\": " << s.stddev_ms << ",\n";
    out << "    \"p50_ms\": " << s.p50_ms << ",\n";
    out << "    \"p95_ms\": " << s.p95_ms << ",\n";
    out << "    \"p99_ms\": " << s.p99_ms << ",\n";
    out << "    \"throughput_gb_s\": " << s.throughput_gb_s << "\n";
    out << "  },\n";
    out << "  \"determinism\": {\n";
    out << "    \"checksum_real\": \"" << to_hex(s.checksum_real) << "\",\n";
    out << "    \"checksum_imag\": \"" << to_hex(s.checksum_imag) << "\"\n";
    out << "  }\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            print_usage();
            return 0;
        }
    }

    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }

    if (!is_power_of_two(cfg.size)) {
        std::cerr << "--size must be a non-zero power of two.\n";
        return 1;
    }
    if (cfg.shots == 0) {
        std::cerr << "--shots must be > 0.\n";
        return 1;
    }

    const size_t rows = cfg.size;
    const size_t cols = cfg.size;
    const size_t total = rows * cols;

    const size_t data_arena_bytes = total * sizeof(float) * 2 + 8192;
    const size_t scratch_arena_bytes = total * sizeof(float) * 2 + 8192;

    basalt::MemoryArena data_arena(data_arena_bytes);
    basalt::kernel::ComplexSoA data(data_arena, total);
    if (!data.real || !data.imag) {
        std::cerr << "Failed to allocate benchmark buffers.\n";
        return 1;
    }

    basalt::MemoryArena scratch(scratch_arena_bytes);

    std::vector<float> input_real(total);
    std::vector<float> input_imag(total);
    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < total; ++i) {
        input_real[i] = dist(rng);
        input_imag[i] = dist(rng);
    }

    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0;
    params.taper_width = 200.0;

    auto run_pipeline_once = [&]() {
        std::memcpy(data.real, input_real.data(), total * sizeof(float));
        std::memcpy(data.imag, input_imag.data(), total * sizeof(float));

        scratch.reset();
        basalt::kernel::fft2d_forward(data.real, data.imag, rows, cols, scratch);
        basalt::filter::apply_fk_filter(data, rows, cols, params);
        scratch.reset();
        basalt::kernel::fft2d_inverse(data.real, data.imag, rows, cols, scratch);
    };

    for (size_t i = 0; i < cfg.warmup; ++i) {
        run_pipeline_once();
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(cfg.shots);

    auto total_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < cfg.shots; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        run_pipeline_once();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end - start;
        latencies_ms.push_back(elapsed_ms.count());
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;

    Stats stats;
    stats.total_seconds = total_elapsed.count();
    stats.bytes_per_shot = static_cast<double>(total) * sizeof(float) * 2.0;
    stats.mean_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
                    static_cast<double>(latencies_ms.size());
    stats.p50_ms = percentile_ms(latencies_ms, 0.50);
    stats.p95_ms = percentile_ms(latencies_ms, 0.95);
    stats.p99_ms = percentile_ms(latencies_ms, 0.99);

    double sq_sum = 0.0;
    for (double v : latencies_ms) {
        double d = v - stats.mean_ms;
        sq_sum += d * d;
    }
    stats.stddev_ms = std::sqrt(sq_sum / static_cast<double>(latencies_ms.size()));
    stats.throughput_gb_s =
        (stats.bytes_per_shot * static_cast<double>(cfg.shots)) / stats.total_seconds / 1e9;

    stats.checksum_real = fnv1a_hash_floats(data.real, total);
    stats.checksum_imag = fnv1a_hash_floats(data.imag, total, stats.checksum_real);

    std::cout << std::fixed << std::setprecision(4)
              << "basalt_bench size=" << cfg.size
              << " shots=" << cfg.shots
              << " warmup=" << cfg.warmup
              << " mean_ms=" << stats.mean_ms
              << " p95_ms=" << stats.p95_ms
              << " p99_ms=" << stats.p99_ms
              << " throughput_gb_s=" << stats.throughput_gb_s
              << "\n";

    if (!cfg.json_out.empty()) {
        std::ofstream json_file(cfg.json_out, std::ios::trunc);
        if (!json_file) {
            std::cerr << "Failed to open JSON output path: " << cfg.json_out << "\n";
            return 1;
        }
        write_json(json_file, cfg, stats);
    } else {
        write_json(std::cout, cfg, stats);
    }

    return 0;
}
