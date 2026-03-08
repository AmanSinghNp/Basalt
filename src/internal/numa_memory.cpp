#include "basalt/internal/numa_memory.hpp"

#include <new>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

#if defined(__linux__)
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#if __has_include(<linux/mempolicy.h>)
#include <linux/mempolicy.h>
#define BASALT_HAS_LINUX_MEMPOLICY 1
#endif
#endif

namespace basalt::internal {

namespace {

constexpr size_t kArenaAlignment = 64;

void aligned_deallocate(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void* aligned_allocate(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, kArenaAlignment);
#else
    const size_t aligned_size = (size + kArenaAlignment - 1) & ~(kArenaAlignment - 1);
    return std::aligned_alloc(kArenaAlignment, aligned_size);
#endif
}

#if defined(__linux__)
void mmap_deallocate(void* ptr) {
    if (!ptr) {
        return;
    }
    const auto* size_ptr = reinterpret_cast<const size_t*>(static_cast<const unsigned char*>(ptr) - sizeof(size_t));
    const size_t mapped_size = *size_ptr;
    void* base = const_cast<size_t*>(size_ptr);
    (void)munmap(base, mapped_size);
}

unsigned long nodemask_for_node(size_t node) {
    return 1UL << (node % (sizeof(unsigned long) * 8));
}

bool bind_range_linux(void* ptr, size_t size, size_t node) {
#if defined(BASALT_HAS_LINUX_MEMPOLICY)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return false;
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned_address = address & ~static_cast<uintptr_t>(page_size - 1);
    const uintptr_t end_address = (address + size + static_cast<uintptr_t>(page_size - 1)) &
                                  ~static_cast<uintptr_t>(page_size - 1);
    const size_t aligned_size = static_cast<size_t>(end_address - aligned_address);
    unsigned long nodemask = nodemask_for_node(node);

    const long result = syscall(
        SYS_mbind,
        reinterpret_cast<void*>(aligned_address),
        aligned_size,
        MPOL_BIND,
        &nodemask,
        sizeof(unsigned long) * 8,
        0UL
    );
    return result == 0;
#else
    (void)ptr;
    (void)size;
    (void)node;
    return false;
#endif
}
#endif

} // namespace

NumaMemoryBlock allocate_numa_local(size_t size,
                                    size_t node,
                                    bool numa_enabled,
                                    MemoryPolicy policy) {
    NumaMemoryBlock block;
    block.size = size;
    block.node = node;
    block.numa_requested = numa_enabled && policy != MemoryPolicy::Default;

#if defined(__linux__)
    if (block.numa_requested && policy == MemoryPolicy::BindWorkerBuffers) {
        const long page_size = sysconf(_SC_PAGESIZE);
        const size_t mapping_size = static_cast<size_t>(page_size > 0 ? page_size : 4096) +
                                    size + kArenaAlignment;
        void* mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping != MAP_FAILED) {
            *reinterpret_cast<size_t*>(mapping) = mapping_size;
            unsigned char* raw = static_cast<unsigned char*>(mapping) + sizeof(size_t);
            uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + (kArenaAlignment - 1)) &
                                ~static_cast<uintptr_t>(kArenaAlignment - 1);
            block.ptr = reinterpret_cast<void*>(aligned);
            block.deallocator = &mmap_deallocate;
            block.numa_bound = bind_range_linux(block.ptr, size, node);
            return block;
        }
    }
#endif

    block.ptr = aligned_allocate(size);
    if (!block.ptr) {
        throw std::bad_alloc();
    }
    block.deallocator = &aligned_deallocate;
    return block;
}

bool bind_existing_range_to_node(const void* ptr,
                                 size_t size,
                                 size_t node,
                                 bool numa_enabled,
                                 MemoryPolicy policy) {
    if (!numa_enabled || policy != MemoryPolicy::BindInputsIfPossible || ptr == nullptr || size == 0) {
        return false;
    }

#if defined(__linux__)
    return bind_range_linux(const_cast<void*>(ptr), size, node);
#else
    (void)node;
    return false;
#endif
}

} // namespace basalt::internal
