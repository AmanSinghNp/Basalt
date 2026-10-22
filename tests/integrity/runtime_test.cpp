#include "basalt/filter/fk_filter.hpp"
#include "basalt/internal/batch_executor.hpp"
#include "basalt/internal/fft2d_auto.hpp"
#include "basalt/internal/numa_topology.hpp"
#include "basalt/internal/task_queue.hpp"
#include "basalt/internal/thread_pool.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include "basalt/kernel/fft2d.hpp"

#include <gtest/gtest.h>

#include "basalt/internal/numa_memory.hpp"
#include "basalt/kernel/cpu_features.hpp"
#include "basalt/kernel/cpu_state.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

basalt::internal::ThreadPoolOptions make_pool_options(size_t threads,
                                                      size_t queue_capacity,
                                                      bool enable_affinity = true,
                                                      bool enable_numa = true) {
    basalt::internal::ThreadPoolOptions options;
    options.thread_count = threads;
    options.queue_capacity = queue_capacity;
    options.enable_affinity = enable_affinity;
    options.enable_numa = enable_numa;
    return options;
}

basalt::filter::FKParams default_fk_params() {
    basalt::filter::FKParams params;
    params.dt = 0.004;
    params.dx = 5.0;
    params.mute_vel_min = -1500.0;
    params.mute_vel_max = 1500.0;
    params.taper_width = 200.0;
    return params;
}

// Temporary directory standing in for /sys/devices/system/node.
class FakeSysfs {
public:
    FakeSysfs() {
        static std::atomic<unsigned> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        m_root = std::filesystem::temp_directory_path() /
                 ("basalt_fake_sysfs_" + std::to_string(stamp) + "_" + std::to_string(counter.fetch_add(1)));
        std::filesystem::create_directories(m_root);
    }
    ~FakeSysfs() {
        std::error_code ec;
        std::filesystem::remove_all(m_root, ec);
    }
    FakeSysfs(const FakeSysfs&) = delete;
    FakeSysfs& operator=(const FakeSysfs&) = delete;

    const std::filesystem::path& root() const { return m_root; }

    void add_node(const std::string& dir_name, const std::string& cpulist, const std::string& distance = "10") {
        const std::filesystem::path dir = m_root / dir_name;
        std::filesystem::create_directories(dir);
        std::ofstream(dir / "cpulist") << cpulist << "\n";
        std::ofstream(dir / "distance") << distance << "\n";
    }

    void add_file(const std::string& name, const std::string& content) {
        std::ofstream(m_root / name) << content << "\n";
    }

private:
    std::filesystem::path m_root;
};

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
    return basalt::kernel::cpu_features().can_use_avx512();
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
    for (const size_t n : {size_t{256}, size_t{1024}, size_t{2048}}) {
        const auto cfg = basalt::internal::default_fft2d_config(n, n);
        EXPECT_EQ(cfg.schedule, basalt::kernel::FFT2DSchedule::RowFirstColumnStrips) << n;
        EXPECT_EQ(cfg.transpose_tile, 128u) << n;
    }
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
    basalt::internal::ThreadPool pool(make_pool_options(4, 128));
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

    // Compare against the direct path computed in this same binary with a relative tolerance,
    // so the test does not depend on the optimisation level or instruction selection.
    float peak = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        peak = std::max({peak, std::fabs(direct_real[i]), std::fabs(direct_imag[i])});
    }
    ASSERT_GT(peak, 0.0f);
    const float tolerance = 1e-4f * peak;
    for (size_t i = 0; i < total; ++i) {
        ASSERT_NEAR(out_real[i], direct_real[i], tolerance) << "index " << i;
        ASSERT_NEAR(out_imag[i], direct_imag[i], tolerance) << "index " << i;
    }
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
    basalt::internal::ThreadPool pool(make_pool_options(4, 128));
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

// ---------------------------------------------------------------------------
// TaskQueue
// ---------------------------------------------------------------------------

TEST(TaskQueueTest, FailedPushLeavesCallerValueIntact) {
    using Fn = std::function<int()>;
    basalt::internal::TaskQueue<Fn> queue(2);
    ASSERT_EQ(queue.capacity(), 2u);

    for (int i = 0; i < 2; ++i) {
        Fn task = [i]() { return i; };
        ASSERT_TRUE(queue.try_push(std::move(task)));
    }

    Fn pending = []() { return 42; };
    EXPECT_FALSE(queue.try_push(std::move(pending)));
    ASSERT_TRUE(static_cast<bool>(pending)) << "a failed push must not consume the task";
    EXPECT_EQ(pending(), 42);

    Fn out;
    ASSERT_TRUE(queue.try_pop(out));
    EXPECT_EQ(out(), 0);
    ASSERT_TRUE(queue.try_push(std::move(pending)));
    ASSERT_TRUE(queue.try_pop(out));
    EXPECT_EQ(out(), 1);
    ASSERT_TRUE(queue.try_pop(out));
    EXPECT_EQ(out(), 42);
    EXPECT_FALSE(queue.try_pop(out));
}

TEST(TaskQueueTest, FailedPushKeepsMoveOnlyValue) {
    basalt::internal::TaskQueue<std::unique_ptr<int>> queue(2);
    ASSERT_TRUE(queue.try_push(std::make_unique<int>(1)));
    ASSERT_TRUE(queue.try_push(std::make_unique<int>(2)));

    auto value = std::make_unique<int>(3);
    EXPECT_FALSE(queue.try_push(std::move(value)));
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, 3);
}

TEST(TaskQueueTest, SizeStaysWithinCapacityUnderContention) {
    basalt::internal::TaskQueue<uint64_t> queue(8);
    constexpr uint64_t kCount = 20000;
    std::atomic<bool> done{false};
    std::atomic<size_t> worst{0};

    std::thread producer([&]() {
        for (uint64_t i = 0; i < kCount; ++i) {
            while (!queue.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });
    std::thread consumer([&]() {
        uint64_t value = 0;
        uint64_t popped = 0;
        while (popped < kCount) {
            if (queue.try_pop(value)) {
                ++popped;
            }
        }
        done.store(true);
    });
    while (!done.load()) {
        const size_t size = queue.size();
        size_t seen = worst.load();
        while (size > seen && !worst.compare_exchange_weak(seen, size)) {
        }
    }
    producer.join();
    consumer.join();

    EXPECT_LE(worst.load(), queue.capacity());
    EXPECT_EQ(queue.size(), 0u);
}

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, TinyQueueCapacityRunsEveryTask) {
    basalt::internal::ThreadPool pool(make_pool_options(4, 2, false, false));
    std::atomic<int> executed{0};
    constexpr int kTasks = 2000;
    for (int i = 0; i < kTasks; ++i) {
        pool.enqueue([&executed]() { executed.fetch_add(1, std::memory_order_relaxed); });
    }
    pool.wait_idle();
    EXPECT_EQ(executed.load(), kTasks);

    const auto stats = pool.stats();
    uint64_t submitted = 0;
    for (uint64_t count : stats.submitted_per_node) {
        submitted += count;
    }
    EXPECT_EQ(submitted, static_cast<uint64_t>(kTasks));
}

TEST(ThreadPoolTest, FiftyTasksThroughCapacityTwoQueue) {
    basalt::internal::ThreadPool pool(make_pool_options(3, 2, false, false));
    std::atomic<int> executed{0};
    for (int i = 0; i < 50; ++i) {
        pool.enqueue([&executed]() {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            executed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    pool.wait_idle();
    EXPECT_EQ(executed.load(), 50);
}

TEST(ThreadPoolTest, RepeatedEnqueueWaitCyclesNeverLoseAWakeup) {
    basalt::internal::ThreadPool pool(make_pool_options(4, 16, false, false));
    std::atomic<int> executed{0};
    for (int round = 0; round < 3000; ++round) {
        pool.enqueue([&executed]() { executed.fetch_add(1, std::memory_order_relaxed); });
        if (round % 3 == 0) {
            // Let the workers go to sleep so the next enqueue has to wake one.
            std::this_thread::yield();
        }
        pool.wait_idle();
    }
    EXPECT_EQ(executed.load(), 3000);
}

TEST(ThreadPoolTest, TaskExceptionIsRethrownFromWaitIdle) {
    for (size_t threads : {size_t{1}, size_t{3}}) {
        SCOPED_TRACE("threads=" + std::to_string(threads));
        basalt::internal::ThreadPool pool(make_pool_options(threads, 16, false, false));
        std::atomic<int> ran{0};
        for (int i = 0; i < 10; ++i) {
            pool.enqueue([&ran, i]() {
                if (i == 2 || i == 7) {
                    throw std::runtime_error("task failed");
                }
                ran.fetch_add(1);
            });
        }
        EXPECT_THROW(pool.wait_idle(), std::runtime_error);
        EXPECT_EQ(ran.load(), 8) << "other tasks must still run";

        // The error is consumed; the pool stays usable.
        EXPECT_NO_THROW(pool.wait_idle());
        pool.enqueue([&ran]() { ran.fetch_add(1); });
        EXPECT_NO_THROW(pool.wait_idle());
        EXPECT_EQ(ran.load(), 9);
    }
}

TEST(ThreadPoolTest, WorkerInitializerExceptionSurfacesFromConstructor) {
    auto options = make_pool_options(4, 16, false, false);
    std::atomic<int> initialized{0};
    options.worker_initializer = [&initialized](size_t index, size_t) {
        if (index == 2) {
            throw std::runtime_error("init failed");
        }
        initialized.fetch_add(1);
    };
    EXPECT_THROW(basalt::internal::ThreadPool pool(options), std::runtime_error);
    EXPECT_EQ(initialized.load(), 3);
}

TEST(ThreadPoolTest, WaitIdleFromInsideTaskThrowsInsteadOfDeadlocking) {
    basalt::internal::ThreadPool pool(make_pool_options(2, 16, false, false));
    std::atomic<bool> got_logic_error{false};
    pool.enqueue([&]() {
        try {
            pool.wait_idle();
        } catch (const std::logic_error&) {
            got_logic_error.store(true);
        }
    });
    pool.wait_idle();
    EXPECT_TRUE(got_logic_error.load());
}

TEST(ThreadPoolTest, WorkersFlushDenormalsToZero) {
    basalt::internal::ThreadPool pool(make_pool_options(2, 16, false, false));
    std::atomic<int> with_ftz{0};
    for (int i = 0; i < 8; ++i) {
        pool.enqueue([&with_ftz]() {
            if (basalt::kernel::check_ftz_daz()) {
                with_ftz.fetch_add(1);
            }
        });
    }
    pool.wait_idle();
    EXPECT_EQ(with_ftz.load(), 8);
}

TEST(ThreadPoolTest, CurrentWorkerIndexIsOnlySetOnWorkers) {
    basalt::internal::ThreadPool pool(make_pool_options(3, 16, false, false));
    EXPECT_EQ(pool.current_worker_index(), basalt::internal::ThreadPool::kNoWorker);
    std::atomic<int> valid{0};
    for (int i = 0; i < 12; ++i) {
        pool.enqueue([&]() {
            if (pool.current_worker_index() < 3) {
                valid.fetch_add(1);
            }
        });
    }
    pool.wait_idle();
    EXPECT_EQ(valid.load(), 12);
}

#if defined(__linux__)
TEST(ThreadPoolTest, AffinityFailuresAreCounted) {
    auto options = make_pool_options(2, 16, true, false);
    options.affinity_cpus = {100000, 100000};  // far beyond CPU_SETSIZE
    basalt::internal::ThreadPool pool(options);
    EXPECT_EQ(pool.affinity_failures(), 2u);
    EXPECT_EQ(pool.stats().affinity_failures, 2u);
}
#endif

TEST(ThreadPoolTest, SparseNodeIdsMapToDenseNodeIndices) {
    FakeSysfs sysfs;
    sysfs.add_node("node0", "0-1");
    sysfs.add_node("node2", "2-3");
    auto options = make_pool_options(4, 64, false, true);
    options.topology_override = basalt::internal::NumaTopology::detect(sysfs.root());
    ASSERT_EQ(options.topology_override->node_count(), 2u);

    basalt::internal::ThreadPool pool(options);
    const auto before = pool.stats();
    ASSERT_EQ(before.node_count, 2u);
    EXPECT_EQ(before.worker_home_node_counts[0], 2u);
    EXPECT_EQ(before.worker_home_node_counts[1], 2u);

    std::atomic<int> executed{0};
    for (size_t i = 0; i < 40; ++i) {
        pool.enqueue([&executed]() { executed.fetch_add(1); }, i);
    }
    pool.wait_idle();
    EXPECT_EQ(executed.load(), 40);
    const auto after = pool.stats();
    EXPECT_EQ(after.submitted_per_node[0], 20u);
    EXPECT_EQ(after.submitted_per_node[1], 20u);
}

TEST(ThreadPoolTest, MoreThanTenNodesAreOrderedNumerically) {
    FakeSysfs sysfs;
    for (int node : {11, 3, 10, 0, 2, 1, 9, 4, 5, 6, 7, 8}) {
        sysfs.add_node("node" + std::to_string(node), std::to_string(node));
    }
    auto options = make_pool_options(12, 64, false, true);
    options.topology_override = basalt::internal::NumaTopology::detect(sysfs.root());
    ASSERT_EQ(options.topology_override->node_count(), 12u);

    basalt::internal::ThreadPool pool(options);
    const auto stats = pool.stats();
    ASSERT_EQ(stats.node_count, 12u);
    for (uint64_t count : stats.worker_home_node_counts) {
        EXPECT_EQ(count, 1u);
    }

    std::atomic<int> executed{0};
    for (size_t i = 0; i < 120; ++i) {
        pool.enqueue([&executed]() { executed.fetch_add(1); }, i);
    }
    pool.wait_idle();
    EXPECT_EQ(executed.load(), 120);
}

// ---------------------------------------------------------------------------
// NumaTopology with a fake sysfs tree
// ---------------------------------------------------------------------------

TEST(NumaTopologyFakeSysfsTest, SparseIdsKeepRealIdsAndDenseIndices) {
    FakeSysfs sysfs;
    sysfs.add_node("node0", "0-1", "10 20");
    sysfs.add_node("node2", "2-3", "20 10");
    sysfs.add_file("possible", "0-2");

    const auto topology = basalt::internal::NumaTopology::detect(sysfs.root());
    EXPECT_TRUE(topology.available());
    EXPECT_FALSE(topology.is_uma());
    ASSERT_EQ(topology.node_count(), 2u);
    EXPECT_EQ(topology.nodes()[0].id, 0u);
    EXPECT_EQ(topology.nodes()[1].id, 2u);
    EXPECT_EQ(topology.nodes()[1].cpus, (std::vector<int>{2, 3}));
    EXPECT_EQ(topology.nodes()[1].distances, (std::vector<int>{20, 10}));

    EXPECT_EQ(topology.node_for_cpu(3), 2u);
    EXPECT_EQ(topology.node_index_for_cpu(3), 1u);
    EXPECT_EQ(topology.node_index_for_cpu(1), 0u);
    EXPECT_EQ(topology.index_for_node_id(2), 1u);
    EXPECT_EQ(topology.index_for_node_id(1), basalt::internal::NumaTopology::npos);
    EXPECT_EQ(topology.node_id_for_index(1), 2u);
    EXPECT_EQ(topology.node_id_for_index(3), 2u);  // taken modulo the node count
    // Unknown cpus fall back to the first node.
    EXPECT_EQ(topology.node_for_cpu(99), 0u);
    EXPECT_EQ(topology.node_index_for_cpu(-1), 0u);
}

TEST(NumaTopologyFakeSysfsTest, NodesSortNumericallyNotLexically) {
    FakeSysfs sysfs;
    for (int node : {10, 2, 0, 1, 11, 3}) {
        sysfs.add_node("node" + std::to_string(node), std::to_string(node));
    }
    const auto topology = basalt::internal::NumaTopology::detect(sysfs.root());
    ASSERT_EQ(topology.node_count(), 6u);
    const std::vector<size_t> expected{0, 1, 2, 3, 10, 11};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(topology.nodes()[i].id, expected[i]) << "index " << i;
    }
    EXPECT_EQ(topology.node_index_for_cpu(10), 4u);
}

TEST(NumaTopologyFakeSysfsTest, EmptyCpuListIsAMemoryOnlyNode) {
    FakeSysfs sysfs;
    sysfs.add_node("node0", "0-3");
    sysfs.add_node("node1", "");
    const auto topology = basalt::internal::NumaTopology::detect(sysfs.root());
    EXPECT_TRUE(topology.available());
    ASSERT_EQ(topology.node_count(), 2u);
    EXPECT_TRUE(topology.nodes()[1].cpus.empty());
    EXPECT_EQ(topology.nodes()[0].cpus.size(), 4u);
}

TEST(NumaTopologyFakeSysfsTest, MalformedCpuListFallsBackToUma) {
    for (const char* bad : {"0-x", "abc", "5-2", "1,,2x", "99999999999999999999", "-3", "1-2-3"}) {
        SCOPED_TRACE(bad);
        FakeSysfs sysfs;
        sysfs.add_node("node0", "0-1");
        sysfs.add_node("node1", bad);
        const auto topology = basalt::internal::NumaTopology::detect(sysfs.root());
        EXPECT_FALSE(topology.available());
        EXPECT_TRUE(topology.is_uma());
        ASSERT_EQ(topology.node_count(), 1u);
        EXPECT_EQ(topology.nodes()[0].id, 0u);
    }
}

TEST(NumaTopologyFakeSysfsTest, MissingOrUnusableRootFallsBackToUma) {
    FakeSysfs sysfs;
    sysfs.add_file("not_a_dir", "x");
    sysfs.add_node("nodeX", "0");       // non-numeric suffix is ignored
    sysfs.add_node("power", "0");       // unrelated directory
    sysfs.add_file("node5", "0");       // a file, not a directory

    const std::vector<std::filesystem::path> roots{
        sysfs.root() / "does_not_exist",
        sysfs.root() / "not_a_dir",
        sysfs.root(),  // no usable nodeN directory
    };
    for (const auto& root : roots) {
        SCOPED_TRACE(root.string());
        basalt::internal::NumaTopology topology;
        ASSERT_NO_THROW(topology = basalt::internal::NumaTopology::detect(root));
        EXPECT_FALSE(topology.available());
        EXPECT_TRUE(topology.is_uma());
        ASSERT_EQ(topology.node_count(), 1u);
    }
}

TEST(NumaTopologyFakeSysfsTest, NonNodeEntriesAreIgnored) {
    FakeSysfs sysfs;
    sysfs.add_node("node0", "0-1");
    sysfs.add_node("nodeX", "oops");
    sysfs.add_node("power", "0");
    sysfs.add_file("online", "0");
    const auto topology = basalt::internal::NumaTopology::detect(sysfs.root());
    EXPECT_TRUE(topology.available());
    EXPECT_TRUE(topology.is_uma());
    ASSERT_EQ(topology.node_count(), 1u);
    EXPECT_EQ(topology.nodes()[0].cpus, (std::vector<int>{0, 1}));
}

// ---------------------------------------------------------------------------
// NUMA memory
// ---------------------------------------------------------------------------

TEST(NumaMemoryTest, MappedBlocksAreUnmappedWithTheirRealSize) {
    using namespace basalt::internal;
    const NumaMemoryDebugStats before = numa_memory_debug_stats();

    constexpr int kRounds = 200;
    constexpr size_t kSize = (1u << 20) + 123;
    for (int i = 0; i < kRounds; ++i) {
        NumaMemoryBlock block = allocate_numa_local(kSize, 0, true, MemoryPolicy::BindWorkerBuffers);
        ASSERT_NE(block.ptr, nullptr);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(block.ptr) % 64, 0u);
        EXPECT_TRUE(block.numa_requested);
        std::memset(block.ptr, 0xA5, block.size);  // the whole request must be usable
    }

    const NumaMemoryDebugStats after = numa_memory_debug_stats();
#if defined(__linux__)
    EXPECT_EQ(after.mapped_blocks_allocated - before.mapped_blocks_allocated, static_cast<uint64_t>(kRounds));
    EXPECT_EQ(after.mapped_blocks_released - before.mapped_blocks_released, static_cast<uint64_t>(kRounds));
    EXPECT_EQ(after.munmap_failures, before.munmap_failures);
    EXPECT_EQ(after.mapped_bytes_live, before.mapped_bytes_live);
#else
    EXPECT_EQ(after.mapped_blocks_allocated, before.mapped_blocks_allocated);
#endif
}

TEST(NumaMemoryTest, BlockIsMoveOnlyAndReleasesOnce) {
    using namespace basalt::internal;
    const NumaMemoryDebugStats before = numa_memory_debug_stats();
    {
        NumaMemoryBlock first = allocate_numa_local(4096, 0, true, MemoryPolicy::BindWorkerBuffers);
        void* ptr = first.ptr;
        NumaMemoryBlock second(std::move(first));
        EXPECT_EQ(first.ptr, nullptr);
        EXPECT_EQ(second.ptr, ptr);

        NumaMemoryBlock third;
        third = std::move(second);
        EXPECT_EQ(second.ptr, nullptr);
        EXPECT_EQ(third.ptr, ptr);
    }
    const NumaMemoryDebugStats after = numa_memory_debug_stats();
    EXPECT_EQ(after.mapped_blocks_allocated - before.mapped_blocks_allocated,
              after.mapped_blocks_released - before.mapped_blocks_released);
    EXPECT_EQ(after.mapped_bytes_live, before.mapped_bytes_live);
}

TEST(NumaMemoryTest, DefaultPolicyUsesPlainAlignedMemory) {
    using namespace basalt::internal;
    const NumaMemoryDebugStats before = numa_memory_debug_stats();
    {
        NumaMemoryBlock block = allocate_numa_local(10000, 0, true, MemoryPolicy::Default);
        ASSERT_NE(block.ptr, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(block.ptr) % 64, 0u);
        EXPECT_FALSE(block.numa_requested);
        std::memset(block.ptr, 1, block.size);
    }
    EXPECT_EQ(numa_memory_debug_stats().mapped_blocks_allocated, before.mapped_blocks_allocated);
}

TEST(NumaMemoryTest, LargeAndOutOfRangeNodeIdsDoNotCrash) {
    using namespace basalt::internal;
    for (size_t node : {size_t{63}, size_t{64}, size_t{100}, size_t{1023}, size_t{1024}, size_t{100000}}) {
        SCOPED_TRACE(node);
        NumaMemoryBlock block = allocate_numa_local(8192, node, true, MemoryPolicy::BindWorkerBuffers);
        ASSERT_NE(block.ptr, nullptr);
        std::memset(block.ptr, 0, block.size);
        if (node >= 1024) {
            EXPECT_FALSE(block.numa_bound);
        }
    }
}

TEST(NumaMemoryTest, BindingNeverTouchesPartialPages) {
    using namespace basalt::internal;
    std::vector<unsigned char> buffer(64 * 1024, 0);
    // No whole page lies inside a 100 byte range: nothing may be bound.
    EXPECT_FALSE(bind_existing_range_to_node(buffer.data() + 1, 100, 0, true, MemoryPolicy::BindInputsIfPossible));
    EXPECT_FALSE(bind_existing_range_to_node(buffer.data() + 1, 4000, 0, true, MemoryPolicy::BindInputsIfPossible));
    // Other policies and empty ranges are no-ops.
    EXPECT_FALSE(bind_existing_range_to_node(buffer.data(), buffer.size(), 0, true, MemoryPolicy::Default));
    EXPECT_FALSE(bind_existing_range_to_node(buffer.data(), buffer.size(), 0, false, MemoryPolicy::BindInputsIfPossible));
    EXPECT_FALSE(bind_existing_range_to_node(nullptr, 100, 0, true, MemoryPolicy::BindInputsIfPossible));
    EXPECT_FALSE(bind_existing_range_to_node(buffer.data(), 0, 0, true, MemoryPolicy::BindInputsIfPossible));
}

// ---------------------------------------------------------------------------
// BatchExecutor
// ---------------------------------------------------------------------------

namespace {

basalt::internal::BatchJob make_job(const float* in_real,
                                    const float* in_imag,
                                    float* out_real,
                                    float* out_imag,
                                    size_t rows,
                                    size_t cols) {
    return {in_real, in_imag, out_real, out_imag, rows, cols,
            static_cast<size_t>(-1), default_fk_params(), basalt::kernel::FFT2DConfig{}};
}

basalt::internal::ExecutionOptions make_exec_options(size_t threads, size_t max_dim, size_t queue_capacity) {
    basalt::internal::ExecutionOptions options;
    options.thread_count = threads;
    options.enable_affinity = false;
    options.enable_numa = false;
    options.max_rows = max_dim;
    options.max_cols = max_dim;
    options.queue_capacity = queue_capacity;
    options.simd_mode = basalt::kernel::SimdMode::AVX2;
    return options;
}

} // namespace

TEST(BatchExecutorTest, ManyJobsThroughTinyQueueMatchSingleThread) {
    constexpr size_t dim = 16;
    constexpr size_t total = dim * dim;
    constexpr size_t shots = 400;
    std::vector<float> input_real(total);
    std::vector<float> input_imag(total);
    fill_input(input_real, input_imag);

    std::vector<float> expected_real(total);
    std::vector<float> expected_imag(total);
    {
        basalt::internal::BatchExecutor single(make_exec_options(1, dim, 64));
        single.run({make_job(input_real.data(), input_imag.data(), expected_real.data(), expected_imag.data(), dim, dim)});
    }

    std::vector<float> out_real(shots * total, -1.0f);
    std::vector<float> out_imag(shots * total, -1.0f);
    std::vector<basalt::internal::BatchJob> jobs;
    for (size_t shot = 0; shot < shots; ++shot) {
        jobs.push_back(make_job(input_real.data(), input_imag.data(),
                                out_real.data() + shot * total, out_imag.data() + shot * total, dim, dim));
    }
    basalt::internal::BatchExecutor multi(make_exec_options(4, dim, 2));
    const auto stats = multi.run(jobs);
    EXPECT_EQ(stats.shot_count, shots);
    for (size_t shot = 0; shot < shots; ++shot) {
        ASSERT_EQ(std::memcmp(out_real.data() + shot * total, expected_real.data(), total * sizeof(float)), 0)
            << "shot " << shot;
        ASSERT_EQ(std::memcmp(out_imag.data() + shot * total, expected_imag.data(), total * sizeof(float)), 0)
            << "shot " << shot;
    }
}

TEST(BatchExecutorTest, ThreeThousandJobsWithDefaultQueueCapacity) {
    constexpr size_t dim = 64;
    constexpr size_t total = dim * dim;
    constexpr size_t shots = 3000;
    std::vector<float> input_real(total, 1.0f);
    std::vector<float> input_imag(total, 0.0f);
    std::vector<float> out_real(shots * total);
    std::vector<float> out_imag(shots * total);

    basalt::internal::ExecutionOptions options;  // default queue capacity (1024)
    options.thread_count = 2;
    options.enable_numa = false;
    options.enable_affinity = false;
    options.max_rows = dim;
    options.max_cols = dim;
    options.simd_mode = basalt::kernel::SimdMode::AVX2;
    basalt::internal::BatchExecutor executor(options);

    std::vector<basalt::internal::BatchJob> jobs;
    jobs.reserve(shots);
    for (size_t shot = 0; shot < shots; ++shot) {
        jobs.push_back(make_job(input_real.data(), input_imag.data(),
                                out_real.data() + shot * total, out_imag.data() + shot * total, dim, dim));
    }
    EXPECT_NO_THROW(executor.run(jobs));
}

TEST(BatchExecutorTest, InlineExecutorsDoNotShareStaleBuffers) {
    constexpr size_t dim = 64;
    constexpr size_t total = dim * dim;
    std::vector<float> input_real(total);
    std::vector<float> input_imag(total);
    fill_input(input_real, input_imag);

    std::vector<float> expected_real = input_real;
    std::vector<float> expected_imag = input_imag;
    run_direct_pipeline_safe(expected_real, expected_imag, dim, dim);

    // `big` is created first, `small` second and destroyed: with a thread_local pointer to the
    // most recently created inline executor's buffers, `big` would then run on freed memory
    // that is also too small for the job.
    basalt::internal::BatchExecutor big(make_exec_options(1, dim, 64));
    {
        basalt::internal::BatchExecutor small(make_exec_options(1, 8, 64));
        std::vector<float> small_in(64, 1.0f);
        std::vector<float> small_out(64);
        small.run({make_job(small_in.data(), nullptr, small_out.data(), nullptr, 8, 8)});
    }

    std::vector<float> out_real(total);
    std::vector<float> out_imag(total);
    ASSERT_NO_THROW(big.run({make_job(input_real.data(), input_imag.data(), out_real.data(), out_imag.data(), dim, dim)}));
    for (size_t i = 0; i < total; ++i) {
        ASSERT_NEAR(out_real[i], expected_real[i], 1e-3f) << i;
        ASSERT_NEAR(out_imag[i], expected_imag[i], 1e-3f) << i;
    }
}

TEST(BatchExecutorTest, SequentialInlineExecutorsEachUseTheirOwnBuffers) {
    constexpr size_t dim = 32;
    constexpr size_t total = dim * dim;
    std::vector<float> input_real(total, 0.5f);
    std::vector<float> input_imag(total, 0.25f);
    std::vector<float> out_a(total);
    std::vector<float> out_b(total);
    for (int round = 0; round < 3; ++round) {
        {
            basalt::internal::BatchExecutor a(make_exec_options(1, dim, 8));
            a.run({make_job(input_real.data(), input_imag.data(), out_a.data(), nullptr, dim, dim)});
        }
        basalt::internal::BatchExecutor b(make_exec_options(1, dim, 8));
        b.run({make_job(input_real.data(), input_imag.data(), out_b.data(), nullptr, dim, dim)});
        EXPECT_EQ(std::memcmp(out_a.data(), out_b.data(), total * sizeof(float)), 0);
    }
}

TEST(BatchExecutorTest, NullOutputImagIsOptional) {
    constexpr size_t dim = 32;
    constexpr size_t total = dim * dim;
    std::vector<float> input_real(total);
    std::vector<float> input_imag(total);
    fill_input(input_real, input_imag);

    for (size_t threads : {size_t{1}, size_t{3}}) {
        SCOPED_TRACE(threads);
        std::vector<float> with_real(total);
        std::vector<float> with_imag(total);
        std::vector<float> without_real(total);
        basalt::internal::BatchExecutor executor(make_exec_options(threads, dim, 16));
        executor.run({make_job(input_real.data(), input_imag.data(), with_real.data(), with_imag.data(), dim, dim)});
        ASSERT_NO_THROW(executor.run({make_job(input_real.data(), input_imag.data(), without_real.data(), nullptr, dim, dim)}));
        EXPECT_EQ(std::memcmp(with_real.data(), without_real.data(), total * sizeof(float)), 0);
    }
}

TEST(BatchExecutorTest, InvalidJobsAreReportedNotCrashed) {
    constexpr size_t dim = 16;
    std::vector<float> in(dim * dim, 1.0f);
    std::vector<float> out(dim * dim);
    basalt::internal::BatchExecutor executor(make_exec_options(1, dim, 8));

    EXPECT_THROW(executor.run({make_job(nullptr, nullptr, out.data(), nullptr, dim, dim)}), std::invalid_argument);
    EXPECT_THROW(executor.run({make_job(in.data(), nullptr, nullptr, nullptr, dim, dim)}), std::invalid_argument);
    EXPECT_THROW(executor.run({make_job(in.data(), nullptr, out.data(), nullptr, 0, dim)}), std::invalid_argument);
    EXPECT_THROW(executor.run({make_job(in.data(), nullptr, out.data(), nullptr, dim, 0)}), std::invalid_argument);
    // rows * cols wraps around to a tiny number if it is not checked.
    const size_t huge = (std::numeric_limits<size_t>::max() / 2) + 2;
    EXPECT_THROW(executor.run({make_job(in.data(), nullptr, out.data(), nullptr, huge, 2)}), std::invalid_argument);
    EXPECT_THROW(executor.run({make_job(in.data(), nullptr, out.data(), nullptr, dim * 2, dim)}), std::runtime_error);

    // Still usable afterwards.
    EXPECT_NO_THROW(executor.run({make_job(in.data(), nullptr, out.data(), nullptr, dim, dim)}));
}

TEST(BatchExecutorTest, OversizedOptionsAreRejectedByConstructor) {
    auto options = make_exec_options(1, 8, 8);
    options.max_rows = std::numeric_limits<size_t>::max() / 2;
    options.max_cols = 4;
    EXPECT_THROW(basalt::internal::BatchExecutor executor(options), std::invalid_argument);
    options.thread_count = 3;
    EXPECT_THROW(basalt::internal::BatchExecutor executor(options), std::invalid_argument);
}

TEST(BatchExecutorTest, RemainingJobsAreSkippedAfterFirstFailure) {
    constexpr size_t dim = 16;
    constexpr size_t total = dim * dim;
    std::vector<float> in(total, 1.0f);
    std::vector<float> out(2 * total, 12345.0f);
    basalt::internal::BatchExecutor executor(make_exec_options(1, dim, 8));

    std::vector<basalt::internal::BatchJob> jobs;
    jobs.push_back(make_job(in.data(), nullptr, out.data(), nullptr, 0, dim));  // invalid
    jobs.push_back(make_job(in.data(), nullptr, out.data() + total, nullptr, dim, dim));
    EXPECT_THROW(executor.run(jobs), std::invalid_argument);
    for (size_t i = 0; i < total; ++i) {
        ASSERT_EQ(out[total + i], 12345.0f) << "job after the failure must not run";
    }
}

TEST(BatchExecutorTest, FailureInThreadedRunPropagatesAndExecutorStaysUsable) {
    constexpr size_t dim = 16;
    constexpr size_t total = dim * dim;
    constexpr size_t shots = 200;
    std::vector<float> in(total, 1.0f);
    std::vector<float> out(shots * total);
    basalt::internal::BatchExecutor executor(make_exec_options(4, dim, 2));

    std::vector<basalt::internal::BatchJob> bad_jobs;
    for (size_t shot = 0; shot < shots; ++shot) {
        bad_jobs.push_back(make_job(in.data(), nullptr, out.data() + shot * total, nullptr, dim * 4, dim));
    }
    EXPECT_THROW(executor.run(bad_jobs), std::runtime_error);

    std::vector<basalt::internal::BatchJob> good_jobs;
    for (size_t shot = 0; shot < shots; ++shot) {
        good_jobs.push_back(make_job(in.data(), nullptr, out.data() + shot * total, nullptr, dim, dim));
    }
    EXPECT_NO_THROW(executor.run(good_jobs));
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
