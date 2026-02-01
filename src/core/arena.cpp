#include "basalt/arena.hpp"

#include <cstdlib>
#include <new>
#include <iostream>

namespace basalt {

MemoryArena::MemoryArena(size_t size) : m_total_size(size), m_offset(0) {
    // In a real implementation this would use posix_memalign or VirtualAlloc
    m_memory_block = std::malloc(size); 
    if (!m_memory_block) {
        throw std::bad_alloc();
    }
}

MemoryArena::~MemoryArena() {
    std::free(m_memory_block);
}

void* MemoryArena::allocate(size_t size, size_t alignment) {
    // Simple bump allocator logic (placeholder)
    // Align current offset
    size_t current_addr = reinterpret_cast<size_t>(m_memory_block) + m_offset;
    size_t aligned_addr = (current_addr + (alignment - 1)) & ~(alignment - 1);
    
    size_t padding = aligned_addr - current_addr;
    if (m_offset + padding + size > m_total_size) {
        return nullptr;
    }

    m_offset += padding + size;
    return reinterpret_cast<void*>(aligned_addr);
}

void MemoryArena::reset() {
    m_offset = 0;
}

} // namespace basalt
