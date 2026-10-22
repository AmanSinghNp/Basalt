#include "basalt/internal/thread_pool.hpp"

#include "basalt/kernel/cpu_state.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace basalt::internal {

namespace {

// Identifies the pool (and worker slot) the current thread belongs to. Only worker threads set
// these, and they clear them again before exiting.
thread_local const ThreadPool* t_current_pool = nullptr;
thread_local size_t t_current_worker = ThreadPool::kNoWorker;

constexpr int kIdleSpinIterations = 64;
constexpr auto kReservedWorkPoll = std::chrono::microseconds(50);

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
      m_topology(options.topology_override ? *options.topology_override : NumaTopology::detect()) {
    const size_t worker_count = options.thread_count == 0 ? 1 : options.thread_count;
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

    // Node indices are dense (0 .. node_count-1) and index m_nodes / m_workers_per_node; the
    // kernel's (possibly sparse) node ids only matter when talking to the kernel.
    m_workers_per_node.assign(node_count, 0);
    m_worker_info.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        int assigned_cpu = -1;
        size_t home_node = 0;
        if (!m_options.affinity_cpus.empty()) {
            assigned_cpu = m_options.affinity_cpus[worker % m_options.affinity_cpus.size()];
            home_node = can_use_numa ? m_topology.node_index_for_cpu(assigned_cpu) : 0;
        } else if (can_use_numa) {
            home_node = worker % node_count;
        }
        home_node = std::min(home_node, node_count - 1);
        m_worker_info.push_back(WorkerInfo{worker, home_node, assigned_cpu});
        m_workers_per_node[home_node] += 1;
    }

    try {
        m_workers.reserve(worker_count);
        for (const WorkerInfo& info : m_worker_info) {
            m_workers.emplace_back([this, info]() { worker_main(info); });
        }

        // Wait until every worker has pinned itself and run the initializer, so that
        // initializer failures surface here instead of terminating a detached thread.
        std::unique_lock<std::mutex> lock(m_start_mutex);
        m_start_cv.wait(lock, [&]() { return m_start_done == m_workers.size(); });
    } catch (...) {
        shutdown_workers();
        throw;
    }

    if (m_init_error) {
        const std::exception_ptr error = m_init_error;
        shutdown_workers();
        std::rethrow_exception(error);
    }
}

ThreadPool::~ThreadPool() {
    if (!m_inline_mode) {
        wait_for_idle();
    }
    shutdown_workers();
}

void ThreadPool::shutdown_workers() noexcept {
    {
        std::lock_guard<std::mutex> lock(m_work_mutex);
        m_stop.store(true, std::memory_order_seq_cst);
    }
    m_work_cv.notify_all();
    for (std::thread& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::record_error(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(m_error_mutex);
    if (!m_first_error) {
        m_first_error = std::move(error);
    }
}

void ThreadPool::enqueue(std::function<void()> task, size_t preferred_node) {
    if (m_inline_mode) {
        try {
            task();
        } catch (...) {
            record_error(std::current_exception());
        }
        return;
    }

    const size_t node_count = m_nodes.size();
    const size_t target_node = node_count == 0 ? 0 : (preferred_node % node_count);
    NodeState& state = *m_nodes[target_node];
    m_pending.fetch_add(1, std::memory_order_relaxed);
    m_queued.fetch_add(1, std::memory_order_seq_cst);
    state.submitted.fetch_add(1, std::memory_order_relaxed);

    // try_push leaves `task` untouched when the queue is full, so retrying is safe.
    unsigned attempts = 0;
    while (!state.queue.try_push(std::move(task))) {
        if (++attempts < 128) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }
    notify_work();
}

void ThreadPool::notify_work() {
    // Dekker-style handshake with block_for_work(): the producer raised m_queued before
    // reading m_sleepers, the worker raises m_sleepers before reading m_queued (all seq_cst),
    // so at least one of them observes the other. Taking the mutex before notifying makes
    // sure a worker that is between its check and its wait has actually started waiting.
    if (m_sleepers.load(std::memory_order_seq_cst) == 0) {
        return;
    }
    { std::lock_guard<std::mutex> lock(m_work_mutex); }
    if (m_nodes.size() > 1) {
        // A task may sit on a queue whose own workers are the only ones allowed to take it.
        m_work_cv.notify_all();
    } else {
        m_work_cv.notify_one();
    }
}

void ThreadPool::block_for_work() {
    std::unique_lock<std::mutex> lock(m_work_mutex);
    m_sleepers.fetch_add(1, std::memory_order_seq_cst);
    const size_t queued = m_queued.load(std::memory_order_seq_cst);
    if (m_stop.load(std::memory_order_seq_cst)) {
        // Fall through: the caller re-checks the stop condition.
    } else if (queued == 0) {
        m_work_cv.wait(lock, [&]() {
            return m_queued.load(std::memory_order_seq_cst) != 0 ||
                   m_stop.load(std::memory_order_seq_cst);
        });
    } else {
        // Work exists but this worker was not allowed to take it (it is reserved for the
        // workers of another node) or its producer has not finished pushing yet. Poll briefly.
        m_work_cv.wait_for(lock, kReservedWorkPoll);
    }
    m_sleepers.fetch_sub(1, std::memory_order_seq_cst);
}

void ThreadPool::wait_for_idle() {
    std::unique_lock<std::mutex> lock(m_idle_mutex);
    m_idle_cv.wait(lock, [&]() { return m_pending.load(std::memory_order_acquire) == 0; });
}

void ThreadPool::wait_idle() {
    if (t_current_pool == this) {
        throw std::logic_error("ThreadPool::wait_idle() must not be called from inside a task");
    }

    if (!m_inline_mode) {
        wait_for_idle();
    }

    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        error = std::move(m_first_error);
        m_first_error = nullptr;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

size_t ThreadPool::current_worker_index() const noexcept {
    return t_current_pool == this ? t_current_worker : kNoWorker;
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
        if (info.home_node < result.node_count) {
            result.worker_home_node_counts[info.home_node] += 1;
        }
    }
    for (size_t node = 0; node < m_nodes.size(); ++node) {
        result.submitted_per_node[node] = m_nodes[node]->submitted.load(std::memory_order_relaxed);
        result.executed_per_node[node] = m_nodes[node]->executed.load(std::memory_order_relaxed);
        result.local_queue_executions_per_node[node] = m_nodes[node]->local_executed.load(std::memory_order_relaxed);
        result.remote_steals_from_node[node] = m_nodes[node]->remote_steals.load(std::memory_order_relaxed);
    }
    result.affinity_failures = m_affinity_failures.load(std::memory_order_relaxed);

    return result;
}

void ThreadPool::worker_main(WorkerInfo info) {
    t_current_pool = this;
    t_current_worker = info.index;

    std::exception_ptr error;
    try {
        // MXCSR is per thread, and these threads are owned by the library.
        basalt::kernel::set_flush_to_zero();
        if (m_options.enable_affinity) {
            pin_thread_if_requested(info);
        }
        if (m_options.worker_initializer) {
            m_options.worker_initializer(info.index, info.home_node);
        }
    } catch (...) {
        error = std::current_exception();
    }

    {
        std::lock_guard<std::mutex> lock(m_start_mutex);
        if (error && !m_init_error) {
            m_init_error = error;
        }
        ++m_start_done;
        m_start_cv.notify_all();
    }

    if (!error) {
        worker_loop(info);
    }

    t_current_pool = nullptr;
    t_current_worker = kNoWorker;
}

void ThreadPool::worker_loop(const WorkerInfo& info) {
    std::function<void()> task;
    int idle_spins = 0;
    for (;;) {
        size_t source_node = info.home_node;
        if (!try_pop_for_worker(info, task, source_node)) {
            if (m_stop.load(std::memory_order_seq_cst) &&
                m_queued.load(std::memory_order_seq_cst) == 0) {
                break;
            }
            if (idle_spins < kIdleSpinIterations) {
                ++idle_spins;
                std::this_thread::yield();
            } else {
                block_for_work();
                idle_spins = 0;
            }
            continue;
        }
        idle_spins = 0;
        m_queued.fetch_sub(1, std::memory_order_seq_cst);

        try {
            task();
        } catch (...) {
            record_error(std::current_exception());
        }
        // Destroy the task (and everything it captured) before it counts as finished, so
        // wait_idle() returning implies the captured state is gone.
        task = nullptr;

        NodeState& source = *m_nodes[source_node];
        source.executed.fetch_add(1, std::memory_order_relaxed);
        if (source_node == info.home_node) {
            source.local_executed.fetch_add(1, std::memory_order_relaxed);
        } else {
            source.remote_steals.fetch_add(1, std::memory_order_relaxed);
        }

        if (m_pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(m_idle_mutex);
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
    if (info.assigned_cpu >= CPU_SETSIZE) {
        m_affinity_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(info.assigned_cpu, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        m_affinity_failures.fetch_add(1, std::memory_order_relaxed);
    }
#else
    (void)info;
#endif
}

} // namespace basalt::internal
