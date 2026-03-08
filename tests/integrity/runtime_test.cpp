#include "basalt/filter/fk_filter.hpp"
#include "basalt/internal/batch_executor.hpp"
#include "basalt/internal/fft2d_auto.hpp"
#include "basalt/internal/numa_topology.hpp"
#include "basalt/internal/task_queue.hpp"
#include "basalt/internal/thread_pool.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/fft2d.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <random>
#include <chrono>
#include <thread>
#include <vector>

namespace {

uint64_t hash_floats(const float* data, size_t count, uint64_t seed = 1469598103934665603ULL) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    uint64_t hash = seed;
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void fill_input(std::vector<float>& real, std::vector<float>& imag) {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < real.size(); ++i) {
        real[i] = dist(rng);
        imag[i] = dist(rng);
    }
}

bool runtime_supports_avx512_filter() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

void run_direct_pipeline_safe(std::vector<float>& real,
                              std::vector<float>& imag,
                              size_t rows,
                              size_t cols) {
    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0;
    params.taper_width = 200.0;

    basalt::MemoryArena arena(rows * cols * sizeof(float) * 2 + 8192);
    basalt::kernel::ComplexSoA data(arena, rows * cols);
    std::memcpy(data.real, real.data(), rows * cols * sizeof(float));
    std::memcpy(data.imag, imag.data(), rows * cols * sizeof(float));

    basalt::MemoryArena scratch(rows * cols * sizeof(float) * 2 + 8192);
    const basalt::kernel::FFT2DConfig fft_config = basalt::internal::resolved_fft2d_config(
        basalt::kernel::FFT2DConfig{}, rows, cols
    );
    basalt::kernel::fft2d_forward(data.real, data.imag, rows, cols, scratch, fft_config);
    basalt::filter::apply_fk_filter(data, rows, cols, params, basalt::kernel::SimdMode::AVX2);
    scratch.reset();
    basalt::kernel::fft2d_inverse(data.real, data.imag, rows, cols, scratch, fft_config);

    std::memcpy(real.data(), data.real, rows * cols * sizeof(float));
    std::memcpy(imag.data(), data.imag, rows * cols * sizeof(float));
}

} // namespace

TEST(FFT2DAutoPolicyTest, MatchesLockedTuningTable) {
    const auto small = basalt::internal::default_fft2d_config(256, 256);
    EXPECT_EQ(small.schedule, basalt::kernel::FFT2DSchedule::FourStepRowFirst);
    EXPECT_EQ(small.transpose_tile, 128u);

    const auto medium = basalt::internal::default_fft2d_config(1024, 1024);
    EXPECT_EQ(medium.schedule, basalt::kernel::FFT2DSchedule::LegacyColumnFirst);
    EXPECT_EQ(medium.transpose_tile, 96u);

    const auto large = basalt::internal::default_fft2d_config(2048, 2048);
    EXPECT_EQ(large.schedule, basalt::kernel::FFT2DSchedule::FourStepRowFirst);
    EXPECT_EQ(large.transpose_tile, 96u);
}

TEST(TaskQueueTest, ConcurrentPushPopPreservesCount) {
    basalt::internal::TaskQueue<uint64_t> queue(1024);
    constexpr uint64_t kValuesPerProducer = 5000;
    constexpr size_t kProducerCount = 4;
    constexpr size_t kConsumerCount = 4;
    constexpr uint64_t kExpectedCount = kValuesPerProducer * kProducerCount;
    const uint64_t kExpectedSum = [&]() {
        uint64_t sum = 0;
        for (size_t producer = 0; producer < kProducerCount; ++producer) {
            for (uint64_t i = 1; i <= kValuesPerProducer; ++i) {
                sum += i + static_cast<uint64_t>(producer) * kValuesPerProducer;
            }
        }
        return sum;
    }();

    std::atomic<uint64_t> produced{0};
    std::atomic<uint64_t> consumed{0};
    std::atomic<uint64_t> sum{0};

    std::vector<std::thread> producers;
    for (size_t producer = 0; producer < kProducerCount; ++producer) {
        producers.emplace_back([&, producer]() {
            for (uint64_t i = 1; i <= kValuesPerProducer; ++i) {
                uint64_t value = i + static_cast<uint64_t>(producer) * kValuesPerProducer;
                while (!queue.try_push(value)) {
                    std::this_thread::yield();
                }
                produced.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::vector<std::thread> consumers;
    for (size_t consumer = 0; consumer < kConsumerCount; ++consumer) {
        consumers.emplace_back([&]() {
            uint64_t value = 0;
            while (consumed.load(std::memory_order_relaxed) < kExpectedCount) {
                if (queue.try_pop(value)) {
                    sum.fetch_add(value, std::memory_order_relaxed);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                } else if (produced.load(std::memory_order_relaxed) >= kExpectedCount) {
                    break;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    for (std::thread& producer : producers) {
        producer.join();
    }
    for (std::thread& consumer : consumers) {
        consumer.join();
    }

    EXPECT_EQ(consumed.load(), kExpectedCount);
    EXPECT_EQ(sum.load(), kExpectedSum);
}

TEST(NumaTopologyTest, DetectsAtLeastOneNode) {
    const basalt::internal::NumaTopology topology = basalt::internal::NumaTopology::detect();
    EXPECT_GE(topology.node_count(), 1u);
    EXPECT_FALSE(topology.nodes().empty());
}

TEST(ThreadPoolNodeSelectionTest, WorkerHomeNodeCountsReflectTopology) {
    basalt::internal::ThreadPool pool({4, 128, true, true, {}, {}});
    const auto stats = pool.stats();

    EXPECT_GE(stats.node_count, 1u);
    uint64_t total_workers = 0;
    for (uint64_t count : stats.worker_home_node_counts) {
        total_workers += count;
    }
    EXPECT_EQ(total_workers, 4u);

    if (stats.node_count > 1) {
        size_t non_zero_nodes = 0;
        for (uint64_t count : stats.worker_home_node_counts) {
            non_zero_nodes += count > 0 ? 1u : 0u;
        }
        EXPECT_GT(non_zero_nodes, 1u);
    }
}

TEST(BatchExecutorTest, SingleThreadMatchesDirectPath) {
    constexpr size_t rows = 128;
    constexpr size_t cols = 128;
    constexpr size_t total = rows * cols;

    std::vector<float> input_real(total);
    std::vector<float> input_imag(total);
    fill_input(input_real, input_imag);

    std::vector<float> direct_real = input_real;
    std::vector<float> direct_imag = input_imag;
    run_direct_pipeline_safe(direct_real, direct_imag, rows, cols);

    std::vector<float> out_real(total);
    std::vector<float> out_imag(total);
    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0;
    params.taper_width = 200.0;

    basalt::internal::BatchExecutor executor({
        1, true, false, basalt::internal::MemoryPolicy::BindWorkerBuffers,
        rows, cols, 64, basalt::kernel::SimdMode::AVX2
    });
    executor.run({
        {
            input_real.data(),
            input_imag.data(),
            out_real.data(),
            out_imag.data(),
            rows,
            cols,
            static_cast<size_t>(-1),
            params,
            basalt::kernel::FFT2DConfig{}
        }
    });

    constexpr uint64_t kGoldenReal = 10207506452863427905ULL;
    constexpr uint64_t kGoldenImag = 7712529816265703748ULL;

    EXPECT_EQ(hash_floats(out_real.data(), total), hash_floats(direct_real.data(), total));
    EXPECT_EQ(hash_floats(out_imag.data(), total), hash_floats(direct_imag.data(), total));
    EXPECT_EQ(hash_floats(out_real.data(), total), kGoldenReal);
    EXPECT_EQ(hash_floats(out_imag.data(), total), kGoldenImag);
}

TEST(BatchExecutorTest, MultiThreadMatchesSingleThreadOutputs) {
    constexpr size_t rows = 64;
    constexpr size_t cols = 64;
    constexpr size_t total = rows * cols;
    constexpr size_t shots = 8;

    std::vector<float> input_real(total);
    std::vector<float> input_imag(total);
    fill_input(input_real, input_imag);

    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0;
    params.taper_width = 200.0;

    std::vector<float> single_real(shots * total);
    std::vector<float> single_imag(shots * total);
    std::vector<float> multi_real(shots * total);
    std::vector<float> multi_imag(shots * total);
    std::vector<basalt::internal::BatchJob> jobs;
    jobs.reserve(shots);
    for (size_t shot = 0; shot < shots; ++shot) {
        jobs.push_back({
            input_real.data(),
            input_imag.data(),
            single_real.data() + shot * total,
            single_imag.data() + shot * total,
            rows,
            cols,
            static_cast<size_t>(-1),
            params,
            basalt::kernel::FFT2DConfig{}
        });
    }

    basalt::internal::BatchExecutor single({
        1, true, false, basalt::internal::MemoryPolicy::BindWorkerBuffers,
        rows, cols, 64, basalt::kernel::SimdMode::AVX2
    });
    single.run(jobs);

    jobs.clear();
    for (size_t shot = 0; shot < shots; ++shot) {
        jobs.push_back({
            input_real.data(),
            input_imag.data(),
            multi_real.data() + shot * total,
            multi_imag.data() + shot * total,
            rows,
            cols,
            static_cast<size_t>(-1),
            params,
            basalt::kernel::FFT2DConfig{}
        });
    }

    basalt::internal::BatchExecutor multi({
        4, true, false, basalt::internal::MemoryPolicy::BindWorkerBuffers,
        rows, cols, 128, basalt::kernel::SimdMode::AVX2
    });
    multi.run(jobs);

    EXPECT_EQ(hash_floats(single_real.data(), shots * total), hash_floats(multi_real.data(), shots * total));
    EXPECT_EQ(hash_floats(single_imag.data(), shots * total), hash_floats(multi_imag.data(), shots * total));
}

TEST(BatchExecutorPreferredNodeTest, PreferredNodeCountsAreTracked) {
    constexpr size_t rows = 32;
    constexpr size_t cols = 32;
    constexpr size_t total = rows * cols;
    std::vector<float> input_real(total, 0.5f);
    std::vector<float> input_imag(total, 0.25f);
    std::vector<float> out_real(total * 4);
    std::vector<float> out_imag(total * 4);

    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.taper_width = 100.0;

    basalt::internal::BatchExecutor executor({
        4, true, true, basalt::internal::MemoryPolicy::BindWorkerBuffers,
        rows, cols, 64, basalt::kernel::SimdMode::AVX2
    });

    std::vector<basalt::internal::BatchJob> jobs;
    for (size_t i = 0; i < 4; ++i) {
        jobs.push_back({
            input_real.data(),
            input_imag.data(),
            out_real.data() + i * total,
            out_imag.data() + i * total,
            rows,
            cols,
            i % 2,
            params,
            basalt::kernel::FFT2DConfig{}
        });
    }

    const auto stats = executor.run(jobs);
    ASSERT_FALSE(stats.preferred_node_counts.empty());
    uint64_t total_preferred = 0;
    for (uint64_t count : stats.preferred_node_counts) {
        total_preferred += count;
    }
    EXPECT_EQ(total_preferred, 4u);

    if (stats.thread_pool.node_count > 1) {
        EXPECT_EQ(stats.preferred_node_counts[0], 2u);
        EXPECT_EQ(stats.preferred_node_counts[1], 2u);
    } else {
        EXPECT_EQ(stats.preferred_node_counts[0], 4u);
    }
}

TEST(BatchExecutorRemoteStealTest, RemoteStealsAppearOnlyWhenTopologyAllowsIt) {
    basalt::internal::ThreadPool pool({4, 128, true, true, {}, {}});
    constexpr size_t task_count = 24;
    for (size_t i = 0; i < task_count; ++i) {
        pool.enqueue([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, 0);
    }
    pool.wait_idle();

    const auto stats = pool.stats();
    uint64_t remote_steals = 0;
    for (uint64_t count : stats.remote_steals_from_node) {
        remote_steals += count;
    }

    if (stats.node_count > 1) {
        EXPECT_GT(remote_steals, 0u);
    } else {
        EXPECT_EQ(remote_steals, 0u);
    }
}

TEST(FKFilterSimdModeTest, ExplicitAvx2MatchesAuto) {
    constexpr size_t rows = 64;
    constexpr size_t cols = 64;
    constexpr size_t total = rows * cols;
    basalt::MemoryArena arena1(total * sizeof(float) * 2 + 4096);
    basalt::MemoryArena arena2(total * sizeof(float) * 2 + 4096);
    basalt::kernel::ComplexSoA auto_data(arena1, total);
    basalt::kernel::ComplexSoA avx2_data(arena2, total);

    for (size_t i = 0; i < total; ++i) {
        auto_data.real[i] = static_cast<float>(i % cols) * 0.01f;
        auto_data.imag[i] = static_cast<float>(i % rows) * -0.02f;
        avx2_data.real[i] = auto_data.real[i];
        avx2_data.imag[i] = auto_data.imag[i];
    }

    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0;
    params.taper_width = 150.0;

    basalt::filter::apply_fk_filter(auto_data, rows, cols, params, basalt::kernel::SimdMode::Auto);
    basalt::filter::apply_fk_filter(avx2_data, rows, cols, params, basalt::kernel::SimdMode::AVX2);

    for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(auto_data.real[i], avx2_data.real[i], 1e-5f);
        EXPECT_NEAR(auto_data.imag[i], avx2_data.imag[i], 1e-5f);
    }
}

TEST(FKFilterSimdModeTest, ExplicitAvx512MatchesAvx2) {
    constexpr size_t rows = 64;
    constexpr size_t cols = 64;
    constexpr size_t total = rows * cols;
    basalt::MemoryArena arena1(total * sizeof(float) * 2 + 4096);
    basalt::MemoryArena arena2(total * sizeof(float) * 2 + 4096);
    basalt::kernel::ComplexSoA avx512_data(arena1, total);
    basalt::kernel::ComplexSoA avx2_data(arena2, total);

    for (size_t i = 0; i < total; ++i) {
        avx512_data.real[i] = static_cast<float>(i % cols) * 0.02f;
        avx512_data.imag[i] = static_cast<float>(i % rows) * -0.03f;
        avx2_data.real[i] = avx512_data.real[i];
        avx2_data.imag[i] = avx512_data.imag[i];
    }

    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0;
    params.taper_width = 150.0;

    basalt::filter::apply_fk_filter(avx512_data, rows, cols, params, basalt::kernel::SimdMode::AVX512);
    basalt::filter::apply_fk_filter(avx2_data, rows, cols, params, basalt::kernel::SimdMode::AVX2);

    for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(avx512_data.real[i], avx2_data.real[i], 1e-5f);
        EXPECT_NEAR(avx512_data.imag[i], avx2_data.imag[i], 1e-5f);
    }
}

TEST(FKFilterSimdModeTest, ExplicitAvx512FallsBackWhenUnsupported) {
    if (runtime_supports_avx512_filter()) {
        GTEST_SKIP() << "Host supports AVX-512 filter path.";
    }

    constexpr size_t rows = 32;
    constexpr size_t cols = 32;
    constexpr size_t total = rows * cols;
    basalt::MemoryArena arena1(total * sizeof(float) * 2 + 4096);
    basalt::MemoryArena arena2(total * sizeof(float) * 2 + 4096);
    basalt::kernel::ComplexSoA avx512_data(arena1, total);
    basalt::kernel::ComplexSoA avx2_data(arena2, total);

    for (size_t i = 0; i < total; ++i) {
        avx512_data.real[i] = static_cast<float>(i) * 0.01f;
        avx512_data.imag[i] = static_cast<float>(i) * -0.02f;
        avx2_data.real[i] = avx512_data.real[i];
        avx2_data.imag[i] = avx512_data.imag[i];
    }

    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.taper_width = 120.0;

    basalt::filter::apply_fk_filter(avx512_data, rows, cols, params, basalt::kernel::SimdMode::AVX512);
    basalt::filter::apply_fk_filter(avx2_data, rows, cols, params, basalt::kernel::SimdMode::AVX2);

    EXPECT_EQ(hash_floats(avx512_data.real, total), hash_floats(avx2_data.real, total));
    EXPECT_EQ(hash_floats(avx512_data.imag, total), hash_floats(avx2_data.imag, total));
}
