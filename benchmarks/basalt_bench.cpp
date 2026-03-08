#include "basalt/filter/fk_filter.hpp"
#include "basalt/internal/batch_executor.hpp"
#include "basalt/internal/fft2d_auto.hpp"
#include "basalt/internal/numa_memory.hpp"
#include "basalt/kernel/simd_mode.hpp"

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
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class NumaMode {
    Off,
    Auto
};

struct Config {
    std::vector<size_t> sizes{1024};
    std::vector<size_t> threads{1};
    std::vector<basalt::kernel::SimdMode> simd_modes{basalt::kernel::SimdMode::Auto};
    std::vector<NumaMode> numa_modes{NumaMode::Off};
    std::vector<basalt::internal::MemoryPolicy> memory_policies{basalt::internal::MemoryPolicy::BindWorkerBuffers};
    size_t shots = 64;
    size_t warmup = 8;
    uint32_t seed = 1337;
    std::string json_out;
    std::string csv_out;
};

struct ResultRow {
    size_t size = 0;
    size_t thread_count = 1;
    basalt::kernel::SimdMode simd_mode = basalt::kernel::SimdMode::Auto;
    NumaMode numa_mode = NumaMode::Off;
    double total_seconds = 0.0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double throughput_gb_s = 0.0;
    uint64_t checksum_real = 0;
    uint64_t checksum_imag = 0;
    basalt::kernel::FFT2DSchedule fft_schedule = basalt::kernel::FFT2DSchedule::Auto;
    size_t fft_tile = 128;
    basalt::internal::MemoryPolicy memory_policy = basalt::internal::MemoryPolicy::Default;
    size_t numa_nodes = 1;
    std::vector<uint64_t> preferred_node_counts;
    std::vector<uint64_t> worker_home_node_counts;
    std::vector<uint64_t> local_queue_executions_per_node;
    std::vector<uint64_t> remote_steals_from_node;
};

bool is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

uint64_t fnv1a_hash_bytes(const unsigned char* data, size_t count, uint64_t seed) {
    uint64_t hash = seed;
    for (size_t i = 0; i < count; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t hash_floats(const float* data, size_t count, uint64_t seed = 1469598103934665603ULL) {
    return fnv1a_hash_bytes(reinterpret_cast<const unsigned char*>(data), count * sizeof(float), seed);
}

template <typename T>
bool parse_unsigned_list(const std::string& text, std::vector<T>& out) {
    std::stringstream stream(text);
    std::string token;
    out.clear();
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            const auto value = static_cast<T>(std::stoull(token));
            out.push_back(value);
        } catch (...) {
            return false;
        }
    }
    return !out.empty();
}

bool parse_simd_list(const std::string& text, std::vector<basalt::kernel::SimdMode>& out) {
    std::stringstream stream(text);
    std::string token;
    out.clear();
    while (std::getline(stream, token, ',')) {
        if (token == "auto") {
            out.push_back(basalt::kernel::SimdMode::Auto);
        } else if (token == "avx2") {
            out.push_back(basalt::kernel::SimdMode::AVX2);
        } else if (token == "avx512") {
            out.push_back(basalt::kernel::SimdMode::AVX512);
        } else {
            return false;
        }
    }
    return !out.empty();
}

bool parse_numa_list(const std::string& text, std::vector<NumaMode>& out) {
    std::stringstream stream(text);
    std::string token;
    out.clear();
    while (std::getline(stream, token, ',')) {
        if (token == "off") {
            out.push_back(NumaMode::Off);
        } else if (token == "auto") {
            out.push_back(NumaMode::Auto);
        } else {
            return false;
        }
    }
    return !out.empty();
}

bool parse_memory_policy_list(const std::string& text,
                              std::vector<basalt::internal::MemoryPolicy>& out) {
    std::stringstream stream(text);
    std::string token;
    out.clear();
    while (std::getline(stream, token, ',')) {
        if (token == "default") {
            out.push_back(basalt::internal::MemoryPolicy::Default);
        } else if (token == "bind-worker-buffers") {
            out.push_back(basalt::internal::MemoryPolicy::BindWorkerBuffers);
        } else if (token == "bind-inputs-if-possible") {
            out.push_back(basalt::internal::MemoryPolicy::BindInputsIfPossible);
        } else {
            return false;
        }
    }
    return !out.empty();
}

const char* simd_mode_name(basalt::kernel::SimdMode mode) {
    switch (mode) {
    case basalt::kernel::SimdMode::Auto: return "auto";
    case basalt::kernel::SimdMode::AVX2: return "avx2";
    case basalt::kernel::SimdMode::AVX512: return "avx512";
    }
    return "unknown";
}

const char* numa_mode_name(NumaMode mode) {
    return mode == NumaMode::Auto ? "auto" : "off";
}

const char* memory_policy_name(basalt::internal::MemoryPolicy policy) {
    switch (policy) {
    case basalt::internal::MemoryPolicy::Default: return "default";
    case basalt::internal::MemoryPolicy::BindWorkerBuffers: return "bind-worker-buffers";
    case basalt::internal::MemoryPolicy::BindInputsIfPossible: return "bind-inputs-if-possible";
    }
    return "unknown";
}

const char* schedule_name(basalt::kernel::FFT2DSchedule schedule) {
    switch (schedule) {
    case basalt::kernel::FFT2DSchedule::Auto: return "auto";
    case basalt::kernel::FFT2DSchedule::LegacyColumnFirst: return "legacy-column-first";
    case basalt::kernel::FFT2DSchedule::FourStepRowFirst: return "four-step-row-first";
    }
    return "unknown";
}

std::string join_counts(const std::vector<uint64_t>& values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << '|';
        }
        out << values[i];
    }
    return out.str();
}

void write_json_counts(std::ostream& out, const std::vector<uint64_t>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << ']';
}

double percentile_ms(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    if (values.size() == 1) {
        return values.front();
    }
    const double pos = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double weight = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - weight) + values[hi] * weight;
}

void print_usage() {
    std::cout
        << "Usage: basalt_bench [options]\n"
        << "  --sizes <csv>     Grid sizes. Default: 1024\n"
        << "  --threads <csv>   Thread counts. Default: 1\n"
        << "  --simd <csv>      SIMD modes: auto,avx2,avx512. Default: auto\n"
        << "  --numa <csv>      NUMA modes: off,auto. Default: off\n"
        << "  --memory-policy <csv> Memory policy: default,bind-worker-buffers,bind-inputs-if-possible\n"
        << "  --shots <N>       Timed shots. Default: 64\n"
        << "  --warmup <N>      Warmup shots. Default: 8\n"
        << "  --seed <N>        RNG seed. Default: 1337\n"
        << "  --json-out <path> JSON output path\n"
        << "  --csv-out <path>  CSV output path\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
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
        } else if (arg == "--sizes" || arg == "--size") {
            const char* value = require_value("--sizes");
            if (value == nullptr || !parse_unsigned_list(value, cfg.sizes)) {
                return false;
            }
        } else if (arg == "--threads") {
            const char* value = require_value("--threads");
            if (value == nullptr || !parse_unsigned_list(value, cfg.threads)) {
                return false;
            }
        } else if (arg == "--simd") {
            const char* value = require_value("--simd");
            if (value == nullptr || !parse_simd_list(value, cfg.simd_modes)) {
                return false;
            }
        } else if (arg == "--numa") {
            const char* value = require_value("--numa");
            if (value == nullptr || !parse_numa_list(value, cfg.numa_modes)) {
                return false;
            }
        } else if (arg == "--memory-policy") {
            const char* value = require_value("--memory-policy");
            if (value == nullptr || !parse_memory_policy_list(value, cfg.memory_policies)) {
                return false;
            }
        } else if (arg == "--shots") {
            const char* value = require_value("--shots");
            if (value == nullptr) return false;
            cfg.shots = static_cast<size_t>(std::stoull(value));
        } else if (arg == "--warmup") {
            const char* value = require_value("--warmup");
            if (value == nullptr) return false;
            cfg.warmup = static_cast<size_t>(std::stoull(value));
        } else if (arg == "--seed") {
            const char* value = require_value("--seed");
            if (value == nullptr) return false;
            cfg.seed = static_cast<uint32_t>(std::stoul(value));
        } else if (arg == "--json-out") {
            const char* value = require_value("--json-out");
            if (value == nullptr) return false;
            cfg.json_out = value;
        } else if (arg == "--csv-out") {
            const char* value = require_value("--csv-out");
            if (value == nullptr) return false;
            cfg.csv_out = value;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    for (size_t size : cfg.sizes) {
        if (!is_power_of_two(size)) {
            std::cerr << "All sizes must be powers of two.\n";
            return false;
        }
    }
    return true;
}

ResultRow run_case(size_t size,
                   size_t thread_count,
                   basalt::kernel::SimdMode simd_mode,
                   NumaMode numa_mode,
                   basalt::internal::MemoryPolicy memory_policy,
                   size_t shots,
                   size_t warmup,
                   uint32_t seed) {
    const size_t total = size * size;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input_real(total);
    std::vector<float> input_imag(total);
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

    const basalt::kernel::FFT2DConfig fft_config = basalt::internal::resolved_fft2d_config(
        basalt::kernel::FFT2DConfig{}, size, size
    );

    const size_t chunk_size = std::max<size_t>(1, std::min<size_t>(shots, thread_count <= 1 ? 1 : thread_count * 2));
    auto run_pass = [&](size_t shot_count, bool collect_results, ResultRow& row) {
        basalt::internal::BatchExecutor executor({
            thread_count,
            true,
            numa_mode == NumaMode::Auto,
            memory_policy,
            size,
            size,
            std::max<size_t>(256, chunk_size * 4),
            simd_mode
        });

        std::vector<float> output_real(chunk_size * total);
        std::vector<float> output_imag(chunk_size * total);
        std::vector<double> latencies;
        uint64_t checksum_real = 1469598103934665603ULL;
        uint64_t checksum_imag = 1099511628211ULL;
        basalt::internal::BatchExecutionStats last_stats;

        const auto total_start = std::chrono::high_resolution_clock::now();
        for (size_t begin = 0; begin < shot_count; begin += chunk_size) {
            const size_t current = std::min(chunk_size, shot_count - begin);
            std::vector<basalt::internal::BatchJob> jobs;
            jobs.reserve(current);
            for (size_t slot = 0; slot < current; ++slot) {
                jobs.push_back({
                    input_real.data(),
                    input_imag.data(),
                    output_real.data() + slot * total,
                    output_imag.data() + slot * total,
                    size,
                    size,
                    static_cast<size_t>(-1),
                    params,
                    fft_config
                });
            }

            std::vector<double> chunk_latencies;
            const auto stats = executor.run(jobs, collect_results ? &chunk_latencies : nullptr);
            last_stats = stats;
            if (collect_results) {
                latencies.insert(latencies.end(), chunk_latencies.begin(), chunk_latencies.end());
                for (size_t slot = 0; slot < current; ++slot) {
                    checksum_real = hash_floats(output_real.data() + slot * total, total, checksum_real);
                    checksum_imag = hash_floats(output_imag.data() + slot * total, total, checksum_imag);
                }
            }
        }
        const auto total_end = std::chrono::high_resolution_clock::now();

        if (collect_results) {
            const std::chrono::duration<double> elapsed = total_end - total_start;
            row.total_seconds = elapsed.count();
            row.mean_ms = latencies.empty() ? 0.0
                : std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(latencies.size());
            row.p50_ms = percentile_ms(latencies, 0.50);
            row.p95_ms = percentile_ms(latencies, 0.95);
            row.p99_ms = percentile_ms(latencies, 0.99);
            row.throughput_gb_s = (static_cast<double>(total) * sizeof(float) * 2.0 * static_cast<double>(shot_count))
                                  / row.total_seconds / 1e9;
            row.checksum_real = checksum_real;
            row.checksum_imag = checksum_imag;
            row.numa_nodes = last_stats.thread_pool.node_count;
            row.memory_policy = last_stats.memory_policy;
            row.preferred_node_counts = last_stats.preferred_node_counts;
            row.worker_home_node_counts = last_stats.thread_pool.worker_home_node_counts;
            row.local_queue_executions_per_node = last_stats.thread_pool.local_queue_executions_per_node;
            row.remote_steals_from_node = last_stats.thread_pool.remote_steals_from_node;
        }
        return last_stats.thread_pool;
    };

    ResultRow row;
    row.size = size;
    row.thread_count = thread_count;
    row.simd_mode = simd_mode;
    row.numa_mode = numa_mode;
    row.memory_policy = memory_policy;
    row.fft_schedule = fft_config.schedule;
    row.fft_tile = fft_config.transpose_tile;

    if (warmup > 0) {
        run_pass(warmup, false, row);
    }
    const auto pool_stats = run_pass(shots, true, row);
    row.numa_nodes = pool_stats.node_count;
    return row;
}

void write_csv(std::ostream& out, const std::vector<ResultRow>& rows) {
    out << "size,threads,simd,numa,memory_policy,fft_schedule,fft_tile,total_seconds,mean_ms,p50_ms,p95_ms,p99_ms,throughput_gb_s,checksum_real,checksum_imag,numa_nodes,preferred_node_counts,worker_home_node_counts,local_queue_executions_per_node,remote_steals_from_node\n";
    for (const ResultRow& row : rows) {
        out << row.size << ','
            << row.thread_count << ','
            << simd_mode_name(row.simd_mode) << ','
            << numa_mode_name(row.numa_mode) << ','
            << memory_policy_name(row.memory_policy) << ','
            << schedule_name(row.fft_schedule) << ','
            << row.fft_tile << ','
            << row.total_seconds << ','
            << row.mean_ms << ','
            << row.p50_ms << ','
            << row.p95_ms << ','
            << row.p99_ms << ','
            << row.throughput_gb_s << ','
            << row.checksum_real << ','
            << row.checksum_imag << ','
            << row.numa_nodes << ','
            << join_counts(row.preferred_node_counts) << ','
            << join_counts(row.worker_home_node_counts) << ','
            << join_counts(row.local_queue_executions_per_node) << ','
            << join_counts(row.remote_steals_from_node) << '\n';
    }
}

void write_json(std::ostream& out, const std::vector<ResultRow>& rows) {
    out << std::fixed << std::setprecision(6);
    out << "{\n  \"results\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const ResultRow& row = rows[i];
        out << "    {\n"
            << "      \"size\": " << row.size << ",\n"
            << "      \"threads\": " << row.thread_count << ",\n"
            << "      \"simd\": \"" << simd_mode_name(row.simd_mode) << "\",\n"
            << "      \"numa\": \"" << numa_mode_name(row.numa_mode) << "\",\n"
            << "      \"memory_policy\": \"" << memory_policy_name(row.memory_policy) << "\",\n"
            << "      \"fft_schedule\": \"" << schedule_name(row.fft_schedule) << "\",\n"
            << "      \"fft_tile\": " << row.fft_tile << ",\n"
            << "      \"total_seconds\": " << row.total_seconds << ",\n"
            << "      \"mean_ms\": " << row.mean_ms << ",\n"
            << "      \"p50_ms\": " << row.p50_ms << ",\n"
            << "      \"p95_ms\": " << row.p95_ms << ",\n"
            << "      \"p99_ms\": " << row.p99_ms << ",\n"
            << "      \"throughput_gb_s\": " << row.throughput_gb_s << ",\n"
            << "      \"checksum_real\": " << row.checksum_real << ",\n"
            << "      \"checksum_imag\": " << row.checksum_imag << ",\n"
            << "      \"numa_nodes\": " << row.numa_nodes << ",\n"
            << "      \"preferred_node_counts\": ";
        write_json_counts(out, row.preferred_node_counts);
        out << ",\n"
            << "      \"worker_home_node_counts\": ";
        write_json_counts(out, row.worker_home_node_counts);
        out << ",\n"
            << "      \"local_queue_executions_per_node\": ";
        write_json_counts(out, row.local_queue_executions_per_node);
        out << ",\n"
            << "      \"remote_steals_from_node\": ";
        write_json_counts(out, row.remote_steals_from_node);
        out << "\n"
            << "    }" << (i + 1 == rows.size() ? "" : ",") << "\n";
    }
    out << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }

    std::vector<ResultRow> results;
    for (size_t size : cfg.sizes) {
        for (size_t threads : cfg.threads) {
            for (basalt::kernel::SimdMode simd_mode : cfg.simd_modes) {
                for (NumaMode numa_mode : cfg.numa_modes) {
                    for (basalt::internal::MemoryPolicy memory_policy : cfg.memory_policies) {
                        if (threads == 1 && numa_mode == NumaMode::Auto) {
                            continue;
                        }
                        const ResultRow row = run_case(
                            size, threads, simd_mode, numa_mode, memory_policy, cfg.shots, cfg.warmup, cfg.seed
                        );
                        results.push_back(row);
                        std::cout << std::fixed << std::setprecision(4)
                                  << "size=" << row.size
                                  << " threads=" << row.thread_count
                                  << " simd=" << simd_mode_name(row.simd_mode)
                                  << " numa=" << numa_mode_name(row.numa_mode)
                                  << " policy=" << memory_policy_name(row.memory_policy)
                                  << " fft=" << schedule_name(row.fft_schedule)
                                  << "/" << row.fft_tile
                                  << " mean_ms=" << row.mean_ms
                                  << " p99_ms=" << row.p99_ms
                                  << " throughput_gb_s=" << row.throughput_gb_s
                                  << " local=" << join_counts(row.local_queue_executions_per_node)
                                  << " remote=" << join_counts(row.remote_steals_from_node)
                                  << "\n";
                    }
                }
            }
        }
    }

    if (!cfg.csv_out.empty()) {
        std::ofstream csv(cfg.csv_out, std::ios::trunc);
        write_csv(csv, results);
    } else {
        write_csv(std::cout, results);
    }

    if (!cfg.json_out.empty()) {
        std::ofstream json(cfg.json_out, std::ios::trunc);
        write_json(json, results);
    }

    return 0;
}
