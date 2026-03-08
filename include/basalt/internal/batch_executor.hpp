#pragma once

#include "basalt/arena.hpp"
#include "basalt/filter/fk_filter.hpp"
#include "basalt/internal/numa_memory.hpp"
#include "basalt/internal/thread_pool.hpp"
#include "basalt/kernel/fft2d.hpp"
#include "basalt/kernel/simd_mode.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace basalt::internal {

struct ExecutionOptions {
    size_t thread_count = 1;
    bool enable_affinity = true;
    bool enable_numa = true;
    MemoryPolicy memory_policy = MemoryPolicy::BindWorkerBuffers;
    size_t max_rows = 1024;
    size_t max_cols = 1024;
    size_t queue_capacity = 1024;
    basalt::kernel::SimdMode simd_mode = basalt::kernel::SimdMode::Auto;
};

struct BatchJob {
    const float* input_real = nullptr;
    const float* input_imag = nullptr;
    float* output_real = nullptr;
    float* output_imag = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t preferred_node = static_cast<size_t>(-1);
    basalt::filter::FKParams params;
    basalt::kernel::FFT2DConfig fft_config;
};

struct BatchExecutionStats {
    size_t shot_count = 0;
    MemoryPolicy memory_policy = MemoryPolicy::Default;
    std::vector<uint64_t> preferred_node_counts;
    ThreadPoolStats thread_pool;
};

class BatchExecutor {
public:
    explicit BatchExecutor(const ExecutionOptions& options = {});
    BatchExecutionStats run(const std::vector<BatchJob>& jobs,
                            std::vector<double>* latencies_ms = nullptr);

    struct WorkerBuffers {
        WorkerBuffers(size_t max_elements,
                      size_t home_node,
                      bool numa_enabled,
                      MemoryPolicy memory_policy);

        NumaMemoryBlock data_block;
        NumaMemoryBlock scratch_block;
        basalt::MemoryArena data_arena;
        basalt::MemoryArena scratch_arena;
        basalt::kernel::ComplexSoA data;
        size_t max_elements = 0;
        size_t home_node = 0;
    };

private:
    static void process_job(const BatchJob& job,
                            WorkerBuffers& buffers,
                            basalt::kernel::SimdMode simd_mode);

    ExecutionOptions m_options;
    std::vector<std::unique_ptr<WorkerBuffers>> m_buffers;
    std::unique_ptr<ThreadPool> m_pool;
};

} // namespace basalt::internal
