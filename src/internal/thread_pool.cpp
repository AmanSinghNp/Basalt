#include "basalt/internal/thread_pool.hpp"

#include <algorithm>
#include <chrono>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace basalt::internal {

namespace {

std::vector<int> build_default_affinity(const NumaTopology& topology, size_t worker_count) {
    std::vector<int> affinity;
    affinity.reserve(worker_count);

    if (topology.nodes().empty()) {
        return affinity;
    }

    std::vector<size_t> node_offsets(topology.node_count(), 0);
    while (affinity.size() < worker_count) {
        bool progressed = false;
        for (size_t node = 0; node < topology.node_count() && affinity.size() < worker_count; ++node) {
            const auto& cpus = topology.nodes()[node].cpus;
            if (cpus.empty()) {
                continue;
            }
            affinity.push_back(cpus[node_offsets[node] % cpus.size()]);
            node_offsets[node] += 1;
            progressed = true;
        }
        if (!progressed) {
            break;
        }
    }

    return affinity;
}

} // namespace

ThreadPool::ThreadPool(const ThreadPoolOptions& options)
    : m_options(options),
      m_topology(NumaTopology::detect()) {
    const size_t requested_threads = options.thread_count == 0 ? 1 : options.thread_count;
    const size_t worker_count = requested_threads == 0 ? 1 : requested_threads;
    const bool can_use_numa = options.enable_numa && m_topology.available() && !m_topology.is_uma();
    const size_t node_count = can_use_numa ? m_topology.node_count() : 1;

    m_nodes.reserve(node_count);
    for (size_t node = 0; node < node_count; ++node) {
        m_nodes.push_back(std::make_unique<NodeState>(options.queue_capacity));
    }

    m_inline_mode = worker_count <= 1;
    if (m_inline_mode) {
        return;
    }

    if (m_options.affinity_cpus.empty()) {
        m_options.affinity_cpus = build_default_affinity(m_topology, worker_count);
    }

    m_workers_per_node.assign(node_count, 0);
    m_worker_info.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        int assigned_cpu = -1;
        size_t home_node = 0;
        if (!m_options.affinity_cpus.empty()) {
            assigned_cpu = m_options.affinity_cpus[worker % m_options.affinity_cpus.size()];
            home_node = can_use_numa ? m_topology.node_for_cpu(assigned_cpu) : 0;
        } else if (can_use_numa) {
            home_node = worker % node_count;
        }
        m_worker_info.push_back(WorkerInfo{worker, home_node, assigned_cpu});
        m_workers_per_node[home_node] += 1;
    }

    for (const WorkerInfo& info : m_worker_info) {
        m_workers.emplace_back([this, info]() {
            if (m_options.enable_affinity) {
                pin_thread_if_requested(info);
            }
            if (m_options.worker_initializer) {
                m_options.worker_initializer(info.index, info.home_node);
            }
            worker_loop(info);
        });
    }
}

ThreadPool::~ThreadPool() {
    m_stop.store(true, std::memory_order_relaxed);
    if (!m_inline_mode) {
        wait_idle();
    }
    for (std::thread& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::enqueue(std::function<void()> task, size_t preferred_node) {
    if (m_inline_mode) {
        task();
        return;
    }

    const size_t node_count = m_nodes.size();
    const size_t target_node = node_count == 0 ? 0 : (preferred_node % node_count);
    NodeState& state = *m_nodes[target_node];
    m_pending.fetch_add(1, std::memory_order_relaxed);
    state.submitted.fetch_add(1, std::memory_order_relaxed);

    while (!state.queue.try_push(std::move(task))) {
        std::this_thread::yield();
    }
}

void ThreadPool::wait_idle() {
    if (m_inline_mode) {
        return;
    }

    std::unique_lock<std::mutex> lock(m_idle_mutex);
    m_idle_cv.wait(lock, [&]() {
        if (m_pending.load(std::memory_order_acquire) != 0) {
            return false;
        }
        for (const auto& node : m_nodes) {
            if (node->queue.size() != 0) {
                return false;
            }
        }
        return true;
    });
}

ThreadPoolStats ThreadPool::stats() const {
    ThreadPoolStats result;
    result.node_count = m_nodes.empty() ? 1 : m_nodes.size();
    result.numa_enabled = result.node_count > 1;
    result.worker_home_node_counts.resize(result.node_count, 0);
    result.submitted_per_node.resize(result.node_count, 0);
    result.executed_per_node.resize(result.node_count, 0);
    result.local_queue_executions_per_node.resize(result.node_count, 0);
    result.remote_steals_from_node.resize(result.node_count, 0);

    for (const WorkerInfo& info : m_worker_info) {
        result.worker_home_node_counts[info.home_node] += 1;
    }
    for (size_t node = 0; node < m_nodes.size(); ++node) {
        result.submitted_per_node[node] = m_nodes[node]->submitted.load(std::memory_order_relaxed);
        result.executed_per_node[node] = m_nodes[node]->executed.load(std::memory_order_relaxed);
        result.local_queue_executions_per_node[node] = m_nodes[node]->local_executed.load(std::memory_order_relaxed);
        result.remote_steals_from_node[node] = m_nodes[node]->remote_steals.load(std::memory_order_relaxed);
    }

    return result;
}

void ThreadPool::worker_loop(WorkerInfo info) {
    std::function<void()> task;
    for (;;) {
        size_t source_node = info.home_node;
        if (!try_pop_for_worker(info, task, source_node)) {
            if (m_stop.load(std::memory_order_relaxed) &&
                m_pending.load(std::memory_order_acquire) == 0) {
                break;
            }
            if (m_pending.load(std::memory_order_acquire) == 0) {
                std::unique_lock<std::mutex> lock(m_idle_mutex);
                m_idle_cv.notify_all();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }

        task();
        NodeState& source = *m_nodes[source_node];
        source.executed.fetch_add(1, std::memory_order_relaxed);
        if (source_node == info.home_node) {
            source.local_executed.fetch_add(1, std::memory_order_relaxed);
        } else {
            source.remote_steals.fetch_add(1, std::memory_order_relaxed);
        }

        if (m_pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::unique_lock<std::mutex> lock(m_idle_mutex);
            m_idle_cv.notify_all();
        }
    }
}

bool ThreadPool::try_pop_for_worker(const WorkerInfo& info, std::function<void()>& task, size_t& source_node) {
    source_node = info.home_node;
    if (m_nodes[info.home_node]->queue.try_pop(task)) {
        return true;
    }

    for (size_t node = 0; node < m_nodes.size(); ++node) {
        if (node == info.home_node) {
            continue;
        }
        if (m_nodes[node]->queue.size() <= m_workers_per_node[node]) {
            continue;
        }
        if (m_nodes[node]->queue.try_pop(task)) {
            source_node = node;
            return true;
        }
    }
    return false;
}

void ThreadPool::pin_thread_if_requested(const WorkerInfo& info) {
#if defined(__linux__)
    if (info.assigned_cpu < 0) {
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(info.assigned_cpu, &cpuset);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)info;
#endif
}

} // namespace basalt::internal
