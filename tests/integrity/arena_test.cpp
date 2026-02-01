#include "basalt/arena.hpp"
#include <gtest/gtest.h>

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
    
    // Should be able to allocate again from start (hypothetically)
    // For now just check we can allocate
    void* ptr2 = arena.allocate(512);
    EXPECT_NE(ptr2, nullptr);
}
