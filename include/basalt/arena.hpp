#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace basalt {

/// Bump-pointer arena. Allocations are never freed individually; reset()
/// rewinds the whole arena.
///
/// Failure contract:
///   * allocate() / allocate_array() return nullptr when the arena is
///     exhausted (including requests whose size would overflow). Callers MUST
///     check the result. Higher level helpers such as ComplexSoA throw
///     std::bad_alloc instead.
///   * allocate() throws std::invalid_argument when `alignment` is not a
///     nonzero power of two (programmer error, not exhaustion).
///   * Constructors throw std::bad_alloc when the backing block cannot be
///     obtained or the requested size cannot be represented.
///
/// Hidden cost: every allocation is followed by kAliasPadding bytes of slack
/// (4K-aliasing avoidance), including the last one. An arena sized exactly
/// 2 * N * sizeof(float) therefore cannot hold two arrays of N floats. Use
/// required_bytes() (or ComplexSoA::required_arena_bytes()) to size arenas.
class MemoryArena {
public:
    using DeallocatorFn = void(*)(void*);

    /// Owns a new 64-byte aligned block of `size` bytes. `size == 0` is
    /// allowed and yields an empty arena for which every allocation fails.
    /// Throws std::bad_alloc if the allocation fails or if `size` is so large
    /// that rounding it up to the 64-byte allocation granularity overflows.
    explicit MemoryArena(size_t size);

    /// Adopts an externally allocated block. `memory_block` must be non-null
    /// (std::bad_alloc otherwise); `deallocator` may be null for blocks that
    /// are not owned. The block need not be 64-byte aligned: allocate()
    /// aligns the returned pointers itself, which can cost up to
    /// `alignment - 1` extra bytes at the start.
    MemoryArena(void* memory_block, size_t size, DeallocatorFn deallocator);
    ~MemoryArena();

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&& other) noexcept;
    MemoryArena& operator=(MemoryArena&& other) noexcept;

    /// Returns `size` bytes aligned to `alignment`, or nullptr if the arena
    /// does not have room for the allocation plus kAliasPadding trailing
    /// bytes. Throws std::invalid_argument if alignment is zero or not a
    /// power of two.
    [[nodiscard]] void* allocate(size_t size, size_t alignment = 64);
    void reset();

    // Capacity introspection
    [[nodiscard]] size_t bytes_used() const noexcept { return m_offset; }
    [[nodiscard]] size_t bytes_remaining() const noexcept { return m_total_size - m_offset; }
    [[nodiscard]] size_t total_size() const noexcept { return m_total_size; }
    
    // Padding added after each allocation to avoid 4K aliasing
    static constexpr size_t kAliasPadding = 128;

    /// Exact number of arena bytes needed to satisfy the given allocation
    /// sizes in order, each with `alignment`, for an arena whose base address
    /// is aligned to `alignment` (always true for arenas that own their block
    /// when alignment <= 64). For external blocks add `alignment - 1` slack.
    /// Returns SIZE_MAX if the total is not representable.
    [[nodiscard]] static size_t required_bytes(std::initializer_list<size_t> sizes,
                                               size_t alignment = 64) noexcept {
        constexpr size_t kMax = std::numeric_limits<size_t>::max();
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            return kMax;
        }
        size_t offset = 0;
        for (size_t size : sizes) {
            const size_t pad = (alignment - (offset & (alignment - 1))) & (alignment - 1);
            if (pad > kMax - offset) return kMax;
            offset += pad;
            if (size > kMax - offset) return kMax;
            offset += size;
            if (kAliasPadding > kMax - offset) return kMax;
            offset += kAliasPadding;
        }
        return offset;
    }

    // Typed allocation helper. Returns nullptr on exhaustion or when
    // count * sizeof(T) would overflow size_t.
    template<typename T>
    [[nodiscard]] T* allocate_array(size_t count, size_t alignment = 64) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            return nullptr;
        }
        return static_cast<T*>(allocate(count * sizeof(T), alignment));
    }

private:
    void* m_memory_block = nullptr;
    size_t m_total_size = 0;
    size_t m_offset = 0;
    DeallocatorFn m_deallocator = nullptr;
};

} // namespace basalt
