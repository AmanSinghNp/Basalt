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

    // Capacity introspection
    [[nodiscard]] size_t bytes_used() const noexcept { return m_offset; }
    [[nodiscard]] size_t bytes_remaining() const noexcept { return m_total_size - m_offset; }
    [[nodiscard]] size_t total_size() const noexcept { return m_total_size; }
    
    // Padding added after each allocation to avoid 4K aliasing
    static constexpr size_t kAliasPadding = 128;

    // Typed allocation helper
    template<typename T>
    [[nodiscard]] T* allocate_array(size_t count, size_t alignment = 64) {
        return static_cast<T*>(allocate(count * sizeof(T), alignment));
    }

private:
    void* m_memory_block = nullptr;
    size_t m_total_size = 0;
    size_t m_offset = 0;
};

} // namespace basalt

