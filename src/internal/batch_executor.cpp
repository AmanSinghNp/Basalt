#include "basalt/internal/batch_executor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace basalt::internal {

namespace {

size_t arena_bytes_for_elements(size_t elements) {
    return elements * sizeof(float) * 2 + 8192;
}

// Largest element count for which arena_bytes_for_elements cannot overflow.
constexpr size_t kMaxArenaElements = (std::numeric_limits<size_t>::max() - 8192) / (sizeof(float) * 2);

size_t checked_max_elements(size_t rows, size_t cols) {
    if (rows != 0 && cols > std::numeric_limits<size_t>::max() / rows) {
        throw std::invalid_argument("ExecutionOptions max_rows * max_cols overflows");
    }
    const size_t elements = std::max<size_t>(1, rows * cols);
    if (elements > kMaxArenaElements) {
        throw std::invalid_argument("ExecutionOptions max_rows * max_cols is too large");
    }
    return elements;
}

} // namespace

BatchExecutor::WorkerBuffers::WorkerBuffers(size_t max_elements_in,
                                           size_t home_node_in,
                                           bool numa_enabled,
                                           MemoryPolicy memory_policy)
    : data_block(allocate_numa_local(
          arena_bytes_for_elements(max_elements_in),
          home_node_in,
          numa_enabled,
          memory_policy
      )),
      scratch_block(allocate_numa_local(
          arena_bytes_for_elements(max_elements_in),
          home_node_in,
          numa_enabled,
          memory_policy
      )),
      // The blocks keep ownership of the memory, so the arenas get no deallocator.
      data_arena(data_block.ptr, data_block.size, nullptr),
      scratch_arena(scratch_block.ptr, scratch_block.size, nullptr),
      data(data_arena, max_elements_in),
      max_elements(max_elements_in),
      home_node(home_node_in) {
}

BatchExecutor::BatchExecutor(const ExecutionOptions& options)
    : m_options(options),
      m_topology(NumaTopology::detect()) {
    m_numa_active = m_options.enable_numa && m_topology.available() && !m_topology.is_uma();

    const size_t max_elements = checked_max_elements(m_options.max_rows, m_options.max_cols);
    const size_t worker_slots = std::max<size_t>(1, options.thread_count);
    m_buffers.resize(worker_slots);

    ThreadPoolOptions pool_options;
    pool_options.thread_count = options.thread_count;
    pool_options.queue_capacity = options.queue_capacity;
    pool_options.enable_affinity = options.enable_affinity;
    pool_options.enable_numa = options.enable_numa;
    // Share one topology snapshot so pool node indices and buffer node ids agree.
    pool_options.topology_override = m_topology;
    // Each worker owns its slot in m_buffers; the pool constructor waits for every
    // initializer, which publishes the slots to the thread that later calls run().
    pool_options.worker_initializer = [this, max_elements](size_t worker, size_t home_node) {
        const size_t node_id = m_numa_active ? m_topology.node_id_for_index(home_node) : 0;
        m_buffers[worker] = std::make_unique<WorkerBuffers>(
            max_elements, node_id, m_numa_active, m_options.memory_policy
        );
    };
    m_pool = std::make_unique<ThreadPool>(pool_options);

    if (m_pool->inline_mode()) {
        const size_t node_id = m_numa_active ? m_topology.node_id_for_index(0) : 0;
        m_buffers[0] = std::make_unique<WorkerBuffers>(
            max_elements, node_id, m_numa_active, m_options.memory_policy
        );
    }
}

BatchExecutor::WorkerBuffers& BatchExecutor::buffers_for_current_thread() {
    const size_t worker = m_pool->current_worker_index();
    if (worker != ThreadPool::kNoWorker && worker < m_buffers.size() && m_buffers[worker]) {
        return *m_buffers[worker];
    }
    // Inline mode: jobs run on the caller's thread and use the executor's single buffer set.
    return *m_buffers.front();
}

BatchExecutionStats BatchExecutor::run(const std::vector<BatchJob>& jobs,
                                       std::vector<double>* latencies_ms) {
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;
    if (latencies_ms != nullptr) {
        latencies_ms->assign(jobs.size(), 0.0);
    }
    const size_t node_count = std::max<size_t>(1, m_pool->stats().node_count);
    std::vector<uint64_t> preferred_node_counts(node_count, 0);

    // Queued tasks reference this frame (failed, first_error, error_mutex, latencies_ms), so
    // the frame must not be left while any of them can still run.
    try {
        for (size_t job_index = 0; job_index < jobs.size(); ++job_index) {
            const BatchJob& job = jobs[job_index];
            // Dense node index (modulo the pool's node count), and the kernel id behind it.
            const size_t preferred_node =
                job.preferred_node != std::numeric_limits<size_t>::max()
                    ? (job.preferred_node % node_count)
                    : (job_index % node_count);
            const size_t preferred_node_id = m_numa_active ? m_topology.node_id_for_index(preferred_node) : 0;
            preferred_node_counts[preferred_node] += 1;

            auto task = [&, job, job_index, preferred_node_id]() {
                if (failed.load(std::memory_order_relaxed)) {
                    return;  // an earlier job already failed; do not waste time on the rest
                }
                try {
                    const auto start = std::chrono::steady_clock::now();
                    WorkerBuffers& buffers = buffers_for_current_thread();
                    const size_t total = validate_job(job, buffers.max_elements);
                    const size_t bytes = total * sizeof(float);
                    (void)bind_existing_range_to_node(
                        job.input_real, bytes, preferred_node_id, m_numa_active, m_options.memory_policy
                    );
                    if (job.input_imag != nullptr) {
                        (void)bind_existing_range_to_node(
                            job.input_imag, bytes, preferred_node_id, m_numa_active, m_options.memory_policy
                        );
                    }
                    process_job(job, buffers, m_options.simd_mode);
                    if (latencies_ms != nullptr) {
                        const auto end = std::chrono::steady_clock::now();
                        const std::chrono::duration<double, std::milli> elapsed = end - start;
                        (*latencies_ms)[job_index] = elapsed.count();
                    }
                } catch (...) {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error == nullptr) {
                        first_error = std::current_exception();
                    }
                }
            };

            m_pool->enqueue(std::move(task), preferred_node);
        }
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        try {
            m_pool->wait_idle();
        } catch (...) {
            // The enqueue failure being propagated is the more relevant error.
        }
        throw;
    }

    m_pool->wait_idle();
    if (failed.load(std::memory_order_relaxed) && first_error != nullptr) {
        std::rethrow_exception(first_error);
    }

    BatchExecutionStats stats;
    stats.shot_count = jobs.size();
    stats.memory_policy = m_options.memory_policy;
    stats.preferred_node_counts = std::move(preferred_node_counts);
    stats.thread_pool = m_pool->stats();
    return stats;
}

size_t BatchExecutor::validate_job(const BatchJob& job, size_t max_elements) {
    if (job.input_real == nullptr || job.output_real == nullptr) {
        throw std::invalid_argument("Batch job requires non-null input_real and output_real");
    }
    if (job.rows == 0 || job.cols == 0) {
        throw std::invalid_argument("Batch job has zero rows or columns");
    }
    if (job.cols > std::numeric_limits<size_t>::max() / job.rows) {
        throw std::invalid_argument("Batch job rows * cols overflows");
    }
    const size_t total = job.rows * job.cols;
    if (total > max_elements) {
        throw std::runtime_error("Batch job exceeds worker capacity");
    }
    return total;
}

void BatchExecutor::process_job(const BatchJob& job,
                                WorkerBuffers& buffers,
                                basalt::kernel::SimdMode simd_mode) {
    const size_t total = validate_job(job, buffers.max_elements);

    std::memcpy(buffers.data.real, job.input_real, total * sizeof(float));
    if (job.input_imag != nullptr) {
        std::memcpy(buffers.data.imag, job.input_imag, total * sizeof(float));
    } else {
        std::memset(buffers.data.imag, 0, total * sizeof(float));
    }

    buffers.scratch_arena.reset();
    basalt::kernel::fft2d_forward(
        buffers.data.real, buffers.data.imag,
        job.rows, job.cols,
        buffers.scratch_arena,
        job.fft_config
    );

    basalt::filter::apply_fk_filter(
        buffers.data, job.rows, job.cols, job.params, simd_mode
    );

    buffers.scratch_arena.reset();
    basalt::kernel::fft2d_inverse(
        buffers.data.real, buffers.data.imag,
        job.rows, job.cols,
        buffers.scratch_arena,
        job.fft_config
    );

    std::memcpy(job.output_real, buffers.data.real, total * sizeof(float));
    if (job.output_imag != nullptr) {
        std::memcpy(job.output_imag, buffers.data.imag, total * sizeof(float));
    }
}

} // namespace basalt::internal
