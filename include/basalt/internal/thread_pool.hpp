#pragma once

#include "basalt/internal/numa_topology.hpp"
#include "basalt/internal/task_queue.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace basalt::internal {

struct ThreadPoolOptions {
    size_t thread_count = 1;
    size_t queue_capacity = 1024;
    bool enable_affinity = true;
    bool enable_numa = true;
    std::vector<int> affinity_cpus;
    std::function<void(size_t, size_t)> worker_initializer;
};

struct ThreadPoolStats {
    bool numa_enabled = false;
    size_t node_count = 1;
    std::vector<uint64_t> worker_home_node_counts;
    std::vector<uint64_t> submitted_per_node;
    std::vector<uint64_t> executed_per_node;
    std::vector<uint64_t> local_queue_executions_per_node;
    std::vector<uint64_t> remote_steals_from_node;
};

class ThreadPool {
public:
    explicit ThreadPool(const ThreadPoolOptions& options = {});
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void enqueue(std::function<void()> task, size_t preferred_node = 0);
    void wait_idle();

    [[nodiscard]] bool inline_mode() const noexcept { return m_inline_mode; }
    [[nodiscard]] size_t thread_count() const noexcept { return m_workers.size(); }
    [[nodiscard]] const NumaTopology& topology() const noexcept { return m_topology; }
    [[nodiscard]] ThreadPoolStats stats() const;

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

    void worker_loop(WorkerInfo info);
    bool try_pop_for_worker(const WorkerInfo& info, std::function<void()>& task, size_t& source_node);
    static void pin_thread_if_requested(const WorkerInfo& info);

    ThreadPoolOptions m_options;
    NumaTopology m_topology;
    bool m_inline_mode = true;
    std::vector<std::unique_ptr<NodeState>> m_nodes;
    std::vector<std::thread> m_workers;
    std::vector<WorkerInfo> m_worker_info;
    std::vector<size_t> m_workers_per_node;
    std::atomic<bool> m_stop{false};
    std::atomic<size_t> m_pending{0};
    std::mutex m_idle_mutex;
    std::condition_variable m_idle_cv;
};

} // namespace basalt::internal
