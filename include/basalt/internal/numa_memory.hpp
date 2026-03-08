#pragma once

#include "basalt/arena.hpp"

#include <cstddef>

namespace basalt::internal {

enum class MemoryPolicy {
    Default,
    BindWorkerBuffers,
    BindInputsIfPossible
};

struct NumaMemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    size_t node = 0;
    bool numa_requested = false;
    bool numa_bound = false;
    basalt::MemoryArena::DeallocatorFn deallocator = nullptr;
};

[[nodiscard]] NumaMemoryBlock allocate_numa_local(size_t size,
                                                  size_t node,
                                                  bool numa_enabled,
                                                  MemoryPolicy policy);

[[nodiscard]] bool bind_existing_range_to_node(const void* ptr,
                                               size_t size,
                                               size_t node,
                                               bool numa_enabled,
                                               MemoryPolicy policy);

} // namespace basalt::internal
