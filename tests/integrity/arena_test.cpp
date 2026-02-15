#include "basalt/arena.hpp"
#include <gtest/gtest.h>
#include <cstdint>

TEST(ArenaTest, AllocationBasic) {
    // simple smoke test
    basalt::MemoryArena arena(1024);
    void* ptr = arena.allocate(128);
    EXPECT_NE(ptr, nullptr);
}

TEST(ArenaTest, ResetWorks) {
    basalt::MemoryArena arena(1024);
    void* ptr1 = arena.allocate(512);
    EXPECT_NE(ptr1, nullptr);
    
    arena.reset();
    
    // Should be able to allocate again from start
    void* ptr2 = arena.allocate(512);
    EXPECT_NE(ptr2, nullptr);
}

TEST(ArenaTest, AllocationAlignment64) {
    basalt::MemoryArena arena(4096);
    
    // Allocate with default 64-byte alignment
    void* ptr1 = arena.allocate(100);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 64, 0);
    
    // Allocate again - should still be 64-byte aligned
    void* ptr2 = arena.allocate(100);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 64, 0);
    
    // Allocate with 32-byte alignment
    void* ptr3 = arena.allocate(100, 32);
    EXPECT_NE(ptr3, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr3) % 32, 0);
}

TEST(ArenaTest, OverflowReturnsNullptr) {
    // Capacity for one allocation + padding, but not two
    size_t alloc_size = 200;
    size_t capacity = alloc_size + basalt::MemoryArena::kAliasPadding + 10;
    basalt::MemoryArena arena(capacity);
    
    // First allocation succeeds
    void* ptr1 = arena.allocate(alloc_size);
    EXPECT_NE(ptr1, nullptr);
    
    // Second allocation should fail — not enough space
    void* ptr2 = arena.allocate(alloc_size);
    EXPECT_EQ(ptr2, nullptr);
}

TEST(ArenaTest, ExactCapacityAllocation) {
    // Arena size must include the alias padding
    constexpr size_t alloc_size = 64;
    basalt::MemoryArena arena(alloc_size + basalt::MemoryArena::kAliasPadding);
    
    void* ptr = arena.allocate(alloc_size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
}

TEST(ArenaTest, CapacityIntrospection) {
    basalt::MemoryArena arena(4096);
    EXPECT_EQ(arena.total_size(), 4096u);
    EXPECT_EQ(arena.bytes_used(), 0u);
    EXPECT_EQ(arena.bytes_remaining(), 4096u);
    
    void* p = arena.allocate(100);
    EXPECT_NE(p, nullptr);
    EXPECT_GT(arena.bytes_used(), 0u);
    EXPECT_LT(arena.bytes_remaining(), 4096u);
    EXPECT_EQ(arena.bytes_used() + arena.bytes_remaining(), arena.total_size());
    
    arena.reset();
    EXPECT_EQ(arena.bytes_used(), 0u);
    EXPECT_EQ(arena.bytes_remaining(), 4096u);
}

TEST(ArenaTest, TypedAllocation) {
    basalt::MemoryArena arena(4096);
    
    // Allocate an array of 16 floats (64 bytes = 1 cache line)
    float* floats = arena.allocate_array<float>(16);
    EXPECT_NE(floats, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(floats) % 64, 0);
    
    // Write and read back to verify memory is usable
    for (int i = 0; i < 16; ++i) {
        floats[i] = static_cast<float>(i) * 1.5f;
    }
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(floats[i], static_cast<float>(i) * 1.5f);
    }
}

TEST(ArenaTest, StressMultipleAlignedAllocations) {
    // Allocate 32 separate 64-byte-aligned blocks; all must be aligned
    constexpr size_t NUM_ALLOCS = 32;
    constexpr size_t ALLOC_SIZE = 100;
    // Account for padding in total size
    basalt::MemoryArena arena(NUM_ALLOCS * (256 + basalt::MemoryArena::kAliasPadding));
    
    void* ptrs[NUM_ALLOCS];
    for (size_t i = 0; i < NUM_ALLOCS; ++i) {
        ptrs[i] = arena.allocate(ALLOC_SIZE);
        ASSERT_NE(ptrs[i], nullptr) << "Allocation " << i << " failed";
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptrs[i]) % 64, 0)
            << "Allocation " << i << " is not 64-byte aligned";
    }
    
    // Verify all pointers are distinct
    for (size_t i = 0; i < NUM_ALLOCS; ++i) {
        for (size_t j = i + 1; j < NUM_ALLOCS; ++j) {
            EXPECT_NE(ptrs[i], ptrs[j])
                << "Pointers " << i << " and " << j << " are identical";
        }
    }
}

TEST(ArenaTest, ResetAndReuseAlignment) {
    basalt::MemoryArena arena(4096);
    
    void* ptr_before = arena.allocate(128);
    EXPECT_NE(ptr_before, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr_before) % 64, 0);
    
    arena.reset();
    
    // After reset, first allocation should return same base (or at least be aligned)
    void* ptr_after = arena.allocate(128);
    EXPECT_NE(ptr_after, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr_after) % 64, 0);
    EXPECT_EQ(ptr_before, ptr_after);  // Should reuse same memory
}

