#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt {

class MemoryArena {
public:
    explicit MemoryArena(size_t size);
    ~MemoryArena();

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    [[nodiscard]] void* allocate(size_t size, size_t alignment = 64);
    void reset();

private:
    void* m_memory_block = nullptr;
    size_t m_total_size = 0;
    size_t m_offset = 0;
};

} // namespace basalt
