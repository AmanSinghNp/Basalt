#include "basalt/internal/numa_memory.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

#if defined(__linux__)
#include <cerrno>
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
// Layout of an mmap-backed block:
//
//   base                         aligned - sizeof(MmapHeader)      aligned
//   | ...padding... | MmapHeader { base, mapping_size } | user bytes (64-byte aligned) ...
//
// The header sits immediately before the pointer handed to the caller, so deallocation can
// recover the real base and length no matter how much alignment padding was needed.
struct MmapHeader {
    void* base;
    size_t mapping_size;
};

// Largest node id we pass to the kernel (CONFIG_NODES_SHIFT is at most 10 on Linux).
constexpr size_t kMaxNodeId = 1024;

std::atomic<uint64_t> g_mapped_allocated{0};
std::atomic<uint64_t> g_mapped_released{0};
std::atomic<uint64_t> g_munmap_failures{0};
std::atomic<uint64_t> g_mapped_bytes_allocated{0};
std::atomic<uint64_t> g_mapped_bytes_released{0};

size_t system_page_size() {
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? static_cast<size_t>(page_size) : 4096;
}

void mmap_deallocate(void* ptr) {
    if (!ptr) {
        return;
    }
    MmapHeader header;
    std::memcpy(&header, static_cast<const unsigned char*>(ptr) - sizeof(MmapHeader), sizeof(header));
    if (munmap(header.base, header.mapping_size) != 0) {
        g_munmap_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_mapped_released.fetch_add(1, std::memory_order_relaxed);
    g_mapped_bytes_released.fetch_add(header.mapping_size, std::memory_order_relaxed);
}

// Calls mbind on an already page-aligned range.
bool mbind_pages(void* page_aligned_address, size_t length, size_t node, bool move_pages) {
#if defined(BASALT_HAS_LINUX_MEMPOLICY)
    if (node >= kMaxNodeId || length == 0) {
        return false;
    }
    constexpr size_t kBitsPerWord = sizeof(unsigned long) * 8;
    std::vector<unsigned long> nodemask(node / kBitsPerWord + 1, 0UL);
    nodemask[node / kBitsPerWord] = 1UL << (node % kBitsPerWord);
    // The kernel only reads maxnode - 1 bits, so ask for one more than the mask holds.
    const unsigned long maxnode = static_cast<unsigned long>(nodemask.size() * kBitsPerWord + 1);
    const unsigned long flags = move_pages ? static_cast<unsigned long>(MPOL_MF_MOVE) : 0UL;

    const long result = syscall(
        SYS_mbind,
        page_aligned_address,
        length,
        MPOL_BIND,
        nodemask.data(),
        maxnode,
        flags
    );
    return result == 0;
#else
    (void)page_aligned_address;
    (void)length;
    (void)node;
    (void)move_pages;
    return false;
#endif
}

// Binds only the whole pages that lie completely inside [ptr, ptr + size).
bool bind_range_inward(void* ptr, size_t size, size_t node, bool move_pages) {
    const uintptr_t page = static_cast<uintptr_t>(system_page_size());
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    if (size > std::numeric_limits<uintptr_t>::max() - address) {
        return false;
    }
    const uintptr_t first = (address + page - 1) & ~(page - 1);
    const uintptr_t last = (address + size) & ~(page - 1);
    if (last <= first) {
        return false;
    }
    return mbind_pages(reinterpret_cast<void*>(first), static_cast<size_t>(last - first), node, move_pages);
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
        const size_t page_size = system_page_size();
        const size_t overhead = sizeof(MmapHeader) + kArenaAlignment;
        if (size > std::numeric_limits<size_t>::max() - overhead - page_size) {
            throw std::bad_alloc();
        }
        const size_t mapping_size = (size + overhead + page_size - 1) & ~(page_size - 1);
        void* mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping != MAP_FAILED) {
            const uintptr_t raw = reinterpret_cast<uintptr_t>(mapping) + sizeof(MmapHeader);
            const uintptr_t aligned = (raw + (kArenaAlignment - 1)) & ~static_cast<uintptr_t>(kArenaAlignment - 1);
            const MmapHeader header{mapping, mapping_size};
            std::memcpy(reinterpret_cast<void*>(aligned - sizeof(MmapHeader)), &header, sizeof(header));

            block.ptr = reinterpret_cast<void*>(aligned);
            block.deallocator = &mmap_deallocate;
            g_mapped_allocated.fetch_add(1, std::memory_order_relaxed);
            g_mapped_bytes_allocated.fetch_add(mapping_size, std::memory_order_relaxed);
            // The whole mapping is ours (page aligned), so bind all of it.
            block.numa_bound = mbind_pages(mapping, mapping_size, node, false);
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
                                 MemoryPolicy policy,
                                 bool move_existing_pages) {
    if (!numa_enabled || policy != MemoryPolicy::BindInputsIfPossible || ptr == nullptr || size == 0) {
        return false;
    }

#if defined(__linux__)
    // mbind does not write through the pointer; it only changes the policy of the pages.
    return bind_range_inward(const_cast<void*>(ptr), size, node, move_existing_pages);
#else
    (void)node;
    (void)move_existing_pages;
    return false;
#endif
}

NumaMemoryDebugStats numa_memory_debug_stats() noexcept {
    NumaMemoryDebugStats stats;
#if defined(__linux__)
    // Read releases first so a concurrent allocate/release pair cannot make live go negative.
    const uint64_t released = g_mapped_released.load(std::memory_order_relaxed);
    const uint64_t released_bytes = g_mapped_bytes_released.load(std::memory_order_relaxed);
    stats.mapped_blocks_allocated = g_mapped_allocated.load(std::memory_order_relaxed);
    stats.mapped_blocks_released = released;
    stats.munmap_failures = g_munmap_failures.load(std::memory_order_relaxed);
    const uint64_t allocated_bytes = g_mapped_bytes_allocated.load(std::memory_order_relaxed);
    stats.mapped_bytes_live = allocated_bytes >= released_bytes ? allocated_bytes - released_bytes : 0;
#endif
    return stats;
}

} // namespace basalt::internal
