#include "basalt/arena.hpp"

#include <limits>
#include <new>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h> // _aligned_malloc, _aligned_free
#else
#include <cstdlib> // aligned_alloc, free
#endif

namespace basalt {

namespace {
constexpr size_t ARENA_ALIGNMENT = 64; // Cache line / AVX-512 alignment

void* aligned_allocate(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, ARENA_ALIGNMENT);
#else
    // aligned_alloc requires size to be multiple of alignment.
    // The caller has already rejected sizes for which this would wrap.
    size_t aligned_size = (size + ARENA_ALIGNMENT - 1) & ~(ARENA_ALIGNMENT - 1);
    return std::aligned_alloc(ARENA_ALIGNMENT, aligned_size);
#endif
}

void aligned_deallocate(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}
} // namespace

MemoryArena::MemoryArena(size_t size) : m_total_size(size), m_offset(0), m_deallocator(nullptr) {
    if (size == 0) {
        // Empty arena: no backing block, every allocate() returns nullptr.
        return;
    }
    // Rounding up to the 64-byte granularity must not wrap around.
    if (size > std::numeric_limits<size_t>::max() - (ARENA_ALIGNMENT - 1)) {
        m_total_size = 0;
        throw std::bad_alloc();
    }
    m_memory_block = aligned_allocate(size);
    if (!m_memory_block) {
        m_total_size = 0;
        throw std::bad_alloc();
    }
    m_deallocator = &aligned_deallocate;
}

MemoryArena::MemoryArena(void* memory_block, size_t size, DeallocatorFn deallocator)
    : m_memory_block(memory_block), m_total_size(size), m_offset(0), m_deallocator(deallocator) {
    if (!m_memory_block) {
        throw std::bad_alloc();
    }
}

MemoryArena::~MemoryArena() {
    if (m_deallocator && m_memory_block) {
        m_deallocator(m_memory_block);
    }
}

MemoryArena::MemoryArena(MemoryArena&& other) noexcept
    : m_memory_block(other.m_memory_block), m_total_size(other.m_total_size),
      m_offset(other.m_offset), m_deallocator(other.m_deallocator) {
    other.m_memory_block = nullptr;
    other.m_total_size = 0;
    other.m_offset = 0;
    other.m_deallocator = nullptr;
}

MemoryArena& MemoryArena::operator=(MemoryArena&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (m_deallocator && m_memory_block) {
        m_deallocator(m_memory_block);
    }

    m_memory_block = other.m_memory_block;
    m_total_size = other.m_total_size;
    m_offset = other.m_offset;
    m_deallocator = other.m_deallocator;

    other.m_memory_block = nullptr;
    other.m_total_size = 0;
    other.m_offset = 0;
    other.m_deallocator = nullptr;
    return *this;
}

void* MemoryArena::allocate(size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(
            "MemoryArena::allocate: alignment must be a nonzero power of two");
    }

    // Align current offset
    const uintptr_t current_ptr = reinterpret_cast<uintptr_t>(m_memory_block) + m_offset;
    const size_t align_padding =
        static_cast<size_t>((alignment - (current_ptr & (alignment - 1))) & (alignment - 1));

    // 4K alias avoidance: add 128 bytes (2 cache lines) of extra padding *after* the allocation.
    // This ensures that two arrays of size 4096 (or any power of 2) won't land on the same
    // valid cache set index modulo 4096.
    //
    // All checks are written as subtractions from the remaining capacity
    // (m_offset <= m_total_size always holds) so that huge `size` values
    // cannot wrap around.
    size_t remaining = m_total_size - m_offset;
    if (align_padding > remaining) {
        return nullptr;
    }
    remaining -= align_padding;
    if (size > remaining || remaining - size < kAliasPadding) {
        return nullptr;
    }

    m_offset += align_padding + size + kAliasPadding;
    return reinterpret_cast<void*>(current_ptr + align_padding);
}

void MemoryArena::reset() {
    m_offset = 0;
}

void MemoryArena::rewind(size_t mark) {
    if (mark > m_offset) {
        throw std::invalid_argument(
            "MemoryArena::rewind: mark is not a previous mark of this arena");
    }
    m_offset = mark;
}

} // namespace basalt
