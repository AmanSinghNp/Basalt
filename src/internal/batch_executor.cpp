#include "basalt/internal/batch_executor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace basalt::internal {

namespace {

thread_local BatchExecutor::WorkerBuffers* g_worker_buffers = nullptr;
thread_local std::unique_ptr<BatchExecutor::WorkerBuffers> g_owned_worker_buffers;

size_t arena_bytes_for_elements(size_t elements) {
    return elements * sizeof(float) * 2 + 8192;
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
      data_arena(data_block.ptr, data_block.size, data_block.deallocator),
      scratch_arena(scratch_block.ptr, scratch_block.size, scratch_block.deallocator),
      data(data_arena, max_elements_in),
      max_elements(max_elements_in),
      home_node(home_node_in) {
}

BatchExecutor::BatchExecutor(const ExecutionOptions& options)
    : m_options(options) {
    const NumaTopology topology = NumaTopology::detect();
    const bool numa_active = m_options.enable_numa && topology.available() && !topology.is_uma();

    m_pool = std::make_unique<ThreadPool>(ThreadPoolOptions{
        options.thread_count,
        options.queue_capacity,
        options.enable_affinity,
        options.enable_numa,
        {},
        [this, numa_active](size_t, size_t home_node) {
            if (g_worker_buffers == nullptr) {
                const size_t max_elements = std::max<size_t>(1, m_options.max_rows * m_options.max_cols);
                g_owned_worker_buffers = std::make_unique<WorkerBuffers>(
                    max_elements,
                    home_node,
                    numa_active,
                    m_options.memory_policy
                );
                g_worker_buffers = g_owned_worker_buffers.get();
            }
        }
    });
    if (m_pool->inline_mode()) {
        const size_t max_elements = std::max<size_t>(1, m_options.max_rows * m_options.max_cols);
        m_buffers.push_back(std::make_unique<WorkerBuffers>(
            max_elements,
            0,
            numa_active,
            m_options.memory_policy
        ));
        g_worker_buffers = m_buffers.front().get();
    }
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

    for (size_t job_index = 0; job_index < jobs.size(); ++job_index) {
        const BatchJob& job = jobs[job_index];
        const size_t preferred_node =
            job.preferred_node != std::numeric_limits<size_t>::max()
                ? (job.preferred_node % node_count)
                : (job_index % node_count);
        preferred_node_counts[preferred_node] += 1;

        auto task = [&, job, job_index, preferred_node]() {
            try {
                const auto start = std::chrono::high_resolution_clock::now();
                WorkerBuffers* buffers = g_worker_buffers;
                if (buffers == nullptr) {
                    buffers = m_buffers.front().get();
                }
                const size_t bytes = job.rows * job.cols * sizeof(float);
                const bool numa_enabled = m_options.enable_numa && !m_pool->topology().is_uma();
                (void)bind_existing_range_to_node(
                    job.input_real, bytes, preferred_node, numa_enabled, m_options.memory_policy
                );
                if (job.input_imag != nullptr) {
                    (void)bind_existing_range_to_node(
                        job.input_imag, bytes, preferred_node, numa_enabled, m_options.memory_policy
                    );
                }
                process_job(job, *buffers, m_options.simd_mode);
                if (latencies_ms != nullptr) {
                    const auto end = std::chrono::high_resolution_clock::now();
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

void BatchExecutor::process_job(const BatchJob& job,
                                WorkerBuffers& buffers,
                                basalt::kernel::SimdMode simd_mode) {
    const size_t total = job.rows * job.cols;
    if (total == 0 || total > buffers.max_elements) {
        throw std::runtime_error("Batch job exceeds worker capacity");
    }

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
    std::memcpy(job.output_imag, buffers.data.imag, total * sizeof(float));
}

} // namespace basalt::internal
