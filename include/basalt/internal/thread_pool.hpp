#pragma once

#include "basalt/internal/numa_topology.hpp"
#include "basalt/internal/task_queue.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace basalt::internal {

struct ThreadPoolOptions {
    size_t thread_count = 1;
    size_t queue_capacity = 1024;
    bool enable_affinity = true;
    bool enable_numa = true;
    std::vector<int> affinity_cpus;
    /// Called once on every worker thread (worker index, home node index) before it takes
    /// tasks. If it throws, the ThreadPool constructor joins all workers and rethrows the
    /// first exception. Not called in inline mode.
    std::function<void(size_t, size_t)> worker_initializer;
    /// Use this topology instead of probing sysfs (tests, or sharing one detection).
    std::optional<NumaTopology> topology_override;
};

struct ThreadPoolStats {
    bool numa_enabled = false;
    size_t node_count = 1;
    std::vector<uint64_t> worker_home_node_counts;
    std::vector<uint64_t> submitted_per_node;
    std::vector<uint64_t> executed_per_node;
    std::vector<uint64_t> local_queue_executions_per_node;
    std::vector<uint64_t> remote_steals_from_node;
    /// Number of workers whose CPU pinning request failed.
    uint64_t affinity_failures = 0;
};

/// Node arguments and statistics use dense node indices (0 .. node_count-1), not kernel ids.
class ThreadPool {
public:
    static constexpr size_t kNoWorker = static_cast<size_t>(-1);

    explicit ThreadPool(const ThreadPoolOptions& options = {});
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Queues `task` on the queue of `preferred_node` (taken modulo the node count). In
    /// inline mode the task runs immediately on the calling thread. An exception escaping a
    /// task never terminates the process: the first one is stored and rethrown by the next
    /// wait_idle().
    void enqueue(std::function<void()> task, size_t preferred_node = 0);

    /// Blocks until every task enqueued so far has finished, then rethrows (and clears) the
    /// first exception thrown by a task since the previous call. Must NOT be called from
    /// inside a task of the same pool: that would wait for itself, so it throws
    /// std::logic_error instead.
    void wait_idle();

    [[nodiscard]] bool inline_mode() const noexcept { return m_inline_mode; }
    [[nodiscard]] size_t thread_count() const noexcept { return m_workers.size(); }
    [[nodiscard]] const NumaTopology& topology() const noexcept { return m_topology; }
    [[nodiscard]] ThreadPoolStats stats() const;

    /// Index of the calling thread within this pool, or kNoWorker when the caller is not one
    /// of this pool's workers (always the case in inline mode).
    [[nodiscard]] size_t current_worker_index() const noexcept;

    /// Number of workers whose pthread_setaffinity_np request failed (or was out of range).
    [[nodiscard]] uint64_t affinity_failures() const noexcept {
        return m_affinity_failures.load(std::memory_order_relaxed);
    }

private:
    struct WorkerInfo {
        size_t index = 0;
        size_t home_node = 0;
        int assigned_cpu = -1;
    };

    struct NodeState {
        explicit NodeState(size_t queue_capacity)
            : queue(queue_capacity) {}

        TaskQueue<std::function<void()>> queue;
        alignas(64) std::atomic<uint64_t> submitted{0};
        alignas(64) std::atomic<uint64_t> executed{0};
        alignas(64) std::atomic<uint64_t> local_executed{0};
        alignas(64) std::atomic<uint64_t> remote_steals{0};
    };

    void worker_main(WorkerInfo info);
    void worker_loop(const WorkerInfo& info);
    bool try_pop_for_worker(const WorkerInfo& info, std::function<void()>& task, size_t& source_node);
    void pin_thread_if_requested(const WorkerInfo& info);
    void record_error(std::exception_ptr error);
    void notify_work();
    void block_for_work();
    void wait_for_idle();
    void shutdown_workers() noexcept;

    ThreadPoolOptions m_options;
    NumaTopology m_topology;
    bool m_inline_mode = true;
    std::vector<std::unique_ptr<NodeState>> m_nodes;
    std::vector<std::thread> m_workers;
    std::vector<WorkerInfo> m_worker_info;
    std::vector<size_t> m_workers_per_node;
    std::atomic<bool> m_stop{false};

    // Tasks submitted and not yet finished (drives wait_idle).
    std::atomic<size_t> m_pending{0};
    std::mutex m_idle_mutex;
    std::condition_variable m_idle_cv;

    // Tasks submitted and not yet popped, plus workers parked on m_work_cv. m_queued is
    // incremented before the push, so it never under-counts the queue contents.
    std::atomic<size_t> m_queued{0};
    std::atomic<size_t> m_sleepers{0};
    std::mutex m_work_mutex;
    std::condition_variable m_work_cv;

    // Worker start-up handshake (constructor waits until every worker ran its initializer).
    std::mutex m_start_mutex;
    std::condition_variable m_start_cv;
    size_t m_start_done = 0;
    std::exception_ptr m_init_error;

    std::mutex m_error_mutex;
    std::exception_ptr m_first_error;

    std::atomic<uint64_t> m_affinity_failures{0};
};

} // namespace basalt::internal
