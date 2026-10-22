#pragma once

#include "basalt/arena.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::internal {

enum class MemoryPolicy {
    Default,
    BindWorkerBuffers,
    BindInputsIfPossible
};

/// Owning, move-only handle to a block returned by allocate_numa_local().
/// The destructor releases the memory through `deallocator`. Use release() to hand the
/// pointer to another owner (the block then no longer frees it).
struct NumaMemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    /// Kernel node id the block was requested on (not a dense index).
    size_t node = 0;
    bool numa_requested = false;
    bool numa_bound = false;
    basalt::MemoryArena::DeallocatorFn deallocator = nullptr;

    NumaMemoryBlock() = default;
    ~NumaMemoryBlock() { reset(); }

    NumaMemoryBlock(const NumaMemoryBlock&) = delete;
    NumaMemoryBlock& operator=(const NumaMemoryBlock&) = delete;

    NumaMemoryBlock(NumaMemoryBlock&& other) noexcept { take(other); }
    NumaMemoryBlock& operator=(NumaMemoryBlock&& other) noexcept {
        if (this != &other) {
            reset();
            take(other);
        }
        return *this;
    }

    /// Frees the memory (if owned) and clears the block.
    void reset() noexcept {
        if (ptr != nullptr && deallocator != nullptr) {
            deallocator(ptr);
        }
        ptr = nullptr;
        deallocator = nullptr;
        size = 0;
    }

    /// Gives up ownership and returns the pointer; the caller must free it with the
    /// deallocator it read beforehand.
    [[nodiscard]] void* release() noexcept {
        void* released = ptr;
        ptr = nullptr;
        deallocator = nullptr;
        return released;
    }

private:
    void take(NumaMemoryBlock& other) noexcept {
        ptr = other.ptr;
        size = other.size;
        node = other.node;
        numa_requested = other.numa_requested;
        numa_bound = other.numa_bound;
        deallocator = other.deallocator;
        other.ptr = nullptr;
        other.size = 0;
        other.deallocator = nullptr;
    }
};

/// Allocates `size` bytes, 64-byte aligned. With `numa_enabled` and
/// MemoryPolicy::BindWorkerBuffers the block is mmap'd and bound to the kernel NUMA node id
/// `node` (best effort; numa_bound reports success). Node ids >= 1024 are rejected (the block
/// is still returned, unbound). Throws std::bad_alloc on failure.
[[nodiscard]] NumaMemoryBlock allocate_numa_local(size_t size,
                                                  size_t node,
                                                  bool numa_enabled,
                                                  MemoryPolicy policy);

/// Best-effort binding of an existing caller-owned buffer to the kernel NUMA node id `node`
/// (MemoryPolicy::BindInputsIfPossible only).
///
/// mbind works on whole pages, and changing the policy of a page also changes it for every
/// other object living on that page. The range is therefore shrunk INWARD to the whole pages
/// fully contained in [ptr, ptr + size): the unaligned head and tail are left alone. Returns
/// false (and does nothing) when no whole page fits, when binding is not requested or when
/// the kernel refuses.
///
/// With `move_existing_pages` the kernel also migrates pages that were already faulted in
/// (MPOL_MF_MOVE, only pages used solely by this process). It defaults to false because
/// repeatedly migrating a buffer shared by several jobs makes the pages ping-pong.
[[nodiscard]] bool bind_existing_range_to_node(const void* ptr,
                                               size_t size,
                                               size_t node,
                                               bool numa_enabled,
                                               MemoryPolicy policy,
                                               bool move_existing_pages = false);

/// Debug counters for the mmap-backed allocation path (all zero on non-Linux platforms).
struct NumaMemoryDebugStats {
    uint64_t mapped_blocks_allocated = 0;
    uint64_t mapped_blocks_released = 0;
    uint64_t munmap_failures = 0;
    /// Bytes currently mapped (allocated minus released).
    uint64_t mapped_bytes_live = 0;
};

[[nodiscard]] NumaMemoryDebugStats numa_memory_debug_stats() noexcept;

} // namespace basalt::internal
