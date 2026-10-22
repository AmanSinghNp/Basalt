#include "basalt/arena.hpp"
#include "basalt/kernel/complex_soa.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

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

// --- mark / rewind / ArenaScope ---

TEST(ArenaTest, RewindReusesMemoryAfterMark) {
    basalt::MemoryArena arena(4096);
    ASSERT_NE(arena.allocate(100), nullptr);

    const size_t m = arena.mark();
    EXPECT_EQ(m, arena.bytes_used());
    void* first = arena.allocate(256);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(arena.allocate(512), nullptr);
    EXPECT_GT(arena.bytes_used(), m);

    arena.rewind(m);
    EXPECT_EQ(arena.mark(), m);
    EXPECT_EQ(arena.bytes_used(), m);
    EXPECT_EQ(arena.bytes_used() + arena.bytes_remaining(), arena.total_size());

    void* again = arena.allocate(256);
    ASSERT_NE(again, nullptr);
    EXPECT_EQ(again, first);
}

TEST(ArenaTest, RewindToZeroMatchesReset) {
    basalt::MemoryArena arena(1024);
    const size_t zero = arena.mark();
    EXPECT_EQ(zero, 0u);
    void* a = arena.allocate(64);
    ASSERT_NE(a, nullptr);
    arena.rewind(zero);
    EXPECT_EQ(arena.bytes_used(), 0u);
    EXPECT_EQ(arena.allocate(64), a);
}

TEST(ArenaTest, RewindToCurrentMarkIsNoOp) {
    basalt::MemoryArena arena(1024);
    ASSERT_NE(arena.allocate(64), nullptr);
    const size_t used = arena.bytes_used();
    arena.rewind(used);
    EXPECT_EQ(arena.bytes_used(), used);
}

TEST(ArenaTest, RewindWithInvalidMarkThrows) {
    basalt::MemoryArena arena(1024);
    ASSERT_NE(arena.allocate(64), nullptr);
    const size_t used = arena.bytes_used();

    // Beyond the current offset: not a previous mark.
    EXPECT_THROW(arena.rewind(used + 1), std::invalid_argument);
    EXPECT_THROW(arena.rewind(std::numeric_limits<size_t>::max()), std::invalid_argument);
    EXPECT_EQ(arena.bytes_used(), used);  // failed rewind changes nothing

    // A mark taken before reset() is invalid after it.
    const size_t high = arena.mark();
    arena.reset();
    EXPECT_THROW(arena.rewind(high), std::invalid_argument);
    EXPECT_EQ(arena.bytes_used(), 0u);
}

TEST(ArenaTest, BytesUsedConsistentAcrossMarkRewind) {
    basalt::MemoryArena arena(4096);
    EXPECT_EQ(arena.bytes_used(), 0u);
    const size_t m0 = arena.mark();
    EXPECT_EQ(m0, 0u);

    ASSERT_NE(arena.allocate(100), nullptr);
    const size_t m1 = arena.mark();
    EXPECT_GE(m1, 100u);
    EXPECT_EQ(arena.bytes_used(), m1);

    ASSERT_NE(arena.allocate(200), nullptr);
    EXPECT_GT(arena.bytes_used(), m1);
    EXPECT_EQ(arena.bytes_used() + arena.bytes_remaining(), arena.total_size());

    arena.rewind(m1);
    EXPECT_EQ(arena.bytes_used(), m1);
    EXPECT_EQ(arena.bytes_used() + arena.bytes_remaining(), arena.total_size());

    arena.rewind(m0);
    EXPECT_EQ(arena.bytes_used(), 0u);
    EXPECT_EQ(arena.bytes_remaining(), arena.total_size());
}

TEST(ArenaTest, ArenaScopeRewindsOnExit) {
    basalt::MemoryArena arena(4096);
    void* kept = arena.allocate(128);
    ASSERT_NE(kept, nullptr);
    const size_t before = arena.bytes_used();

    void* inside = nullptr;
    {
        basalt::ArenaScope scope(arena);
        inside = arena.allocate(256);
        ASSERT_NE(inside, nullptr);
        EXPECT_GT(arena.bytes_used(), before);
    }
    EXPECT_EQ(arena.bytes_used(), before);

    // Memory from the scope is handed out again; memory before it is untouched.
    EXPECT_EQ(arena.allocate(256), inside);
    EXPECT_NE(kept, inside);
}

TEST(ArenaTest, ArenaScopeNests) {
    basalt::MemoryArena arena(8192);
    ASSERT_NE(arena.allocate(64), nullptr);
    const size_t outer_base = arena.bytes_used();

    {
        basalt::ArenaScope outer(arena);
        ASSERT_NE(arena.allocate(128), nullptr);
        const size_t outer_used = arena.bytes_used();
        void* inner_first = nullptr;
        {
            basalt::ArenaScope inner(arena);
            inner_first = arena.allocate(512);
            ASSERT_NE(inner_first, nullptr);
            EXPECT_GT(arena.bytes_used(), outer_used);
        }
        // Inner scope gives back only its own allocation.
        EXPECT_EQ(arena.bytes_used(), outer_used);
        EXPECT_EQ(arena.allocate(512), inner_first);
        arena.rewind(outer_used);
    }
    EXPECT_EQ(arena.bytes_used(), outer_base);
}

TEST(ArenaTest, ArenaScopeWithNoAllocationsIsNoOp) {
    basalt::MemoryArena arena(1024);
    ASSERT_NE(arena.allocate(32), nullptr);
    const size_t used = arena.bytes_used();
    {
        basalt::ArenaScope scope(arena);
    }
    EXPECT_EQ(arena.bytes_used(), used);
}

TEST(ArenaTest, ArenaScopeIsNotCopyOrMovable) {
    static_assert(!std::is_copy_constructible_v<basalt::ArenaScope>);
    static_assert(!std::is_copy_assignable_v<basalt::ArenaScope>);
    static_assert(!std::is_move_constructible_v<basalt::ArenaScope>);
    static_assert(!std::is_move_assignable_v<basalt::ArenaScope>);
    SUCCEED();
}

TEST(ArenaTest, ArenaScopeSurvivesResetInsideScope) {
    // reset() inside a scope is documented as unsupported, but the destructor
    // must not throw or underflow the offset when it happens.
    basalt::MemoryArena arena(1024);
    {
        basalt::ArenaScope scope(arena);
        ASSERT_NE(arena.allocate(64), nullptr);
        arena.reset();
    }
    EXPECT_EQ(arena.bytes_used(), 0u);
}


// --- Overflow / validation regressions ---

TEST(ArenaTest, HugeAllocationSizesReturnNullptr) {
    basalt::MemoryArena arena(4096);
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    // These used to wrap in `offset + padding + size + kAliasPadding`.
    EXPECT_EQ(arena.allocate(kMax), nullptr);
    EXPECT_EQ(arena.allocate(kMax - 100), nullptr);
    EXPECT_EQ(arena.allocate(kMax - 127), nullptr);
    EXPECT_EQ(arena.allocate(kMax - 200, 16), nullptr);
    EXPECT_EQ(arena.allocate(4096), nullptr);
    // A failed allocation must not consume anything.
    EXPECT_EQ(arena.bytes_used(), 0u);
    EXPECT_NE(arena.allocate(64), nullptr);
}

TEST(ArenaTest, HugeAllocationAfterPartialUseReturnsNullptr) {
    basalt::MemoryArena arena(4096);
    ASSERT_NE(arena.allocate(100), nullptr);
    const size_t used = arena.bytes_used();
    EXPECT_EQ(arena.allocate(std::numeric_limits<size_t>::max() - used), nullptr);
    EXPECT_EQ(arena.bytes_used(), used);
}

TEST(ArenaTest, AllocateArrayCountOverflowReturnsNullptr) {
    basalt::MemoryArena arena(4096);
    // count * sizeof(float) wraps to 4 for this count.
    EXPECT_EQ(arena.allocate_array<float>((size_t{1} << 62) + 1), nullptr);
    EXPECT_EQ(arena.allocate_array<double>((size_t{1} << 61) + 1), nullptr);
    EXPECT_EQ(arena.allocate_array<float>(std::numeric_limits<size_t>::max()), nullptr);
    EXPECT_EQ(arena.bytes_used(), 0u);
}

TEST(ArenaTest, ConstructorRejectsSizesThatOverflowRounding) {
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    EXPECT_THROW(basalt::MemoryArena{kMax}, std::bad_alloc);
    EXPECT_THROW(basalt::MemoryArena{kMax - 10}, std::bad_alloc);
    EXPECT_THROW(basalt::MemoryArena{kMax - 62}, std::bad_alloc);
}

TEST(ArenaTest, ZeroSizeArenaIsEmpty) {
    basalt::MemoryArena arena(0);
    EXPECT_EQ(arena.total_size(), 0u);
    EXPECT_EQ(arena.bytes_remaining(), 0u);
    EXPECT_EQ(arena.allocate(1), nullptr);
    EXPECT_EQ(arena.allocate(0), nullptr);
}

TEST(ArenaTest, InvalidAlignmentThrows) {
    basalt::MemoryArena arena(4096);
    EXPECT_THROW((void)arena.allocate(16, 0), std::invalid_argument);
    EXPECT_THROW((void)arena.allocate(16, 48), std::invalid_argument);
    EXPECT_THROW((void)arena.allocate(16, 3), std::invalid_argument);
    EXPECT_EQ(arena.bytes_used(), 0u);

    // Valid powers of two, smaller and larger than 64
    for (size_t align : {size_t{1}, size_t{2}, size_t{8}, size_t{16}, size_t{128}, size_t{256}}) {
        void* p = arena.allocate(8, align);
        ASSERT_NE(p, nullptr) << "alignment " << align;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % align, 0u) << "alignment " << align;
    }
}

TEST(ArenaTest, UnsatisfiableAlignmentReturnsNullptr) {
    basalt::MemoryArena arena(256);
    // Alignment far larger than the arena cannot be satisfied.
    EXPECT_EQ(arena.allocate(8, size_t{1} << 40), nullptr);
}

TEST(ArenaTest, RequiredBytesIsExact) {
    constexpr size_t kA = 1000;
    constexpr size_t kB = 36;
    const size_t need = basalt::MemoryArena::required_bytes({kA, kB});

    {
        basalt::MemoryArena arena(need);
        EXPECT_NE(arena.allocate(kA), nullptr);
        EXPECT_NE(arena.allocate(kB), nullptr);
    }
    {
        basalt::MemoryArena arena(need - 1);
        EXPECT_NE(arena.allocate(kA), nullptr);
        EXPECT_EQ(arena.allocate(kB), nullptr);
    }
    EXPECT_EQ(basalt::MemoryArena::required_bytes({std::numeric_limits<size_t>::max()}),
              std::numeric_limits<size_t>::max());
    EXPECT_EQ(basalt::MemoryArena::required_bytes({8}, 3), std::numeric_limits<size_t>::max());
}

TEST(ArenaTest, ExactlyTwoArraysArenaFailsBecauseOfAliasPadding) {
    // Documented hidden cost: 2 * N * sizeof(float) is not enough for two arrays.
    constexpr size_t n = 256;
    basalt::MemoryArena arena(2 * n * sizeof(float));
    EXPECT_NE(arena.allocate_array<float>(n), nullptr);
    EXPECT_EQ(arena.allocate_array<float>(n), nullptr);
}

// --- ComplexSoA ---

TEST(ComplexSoATest, ThrowsInsteadOfLeavingNullImag) {
    constexpr size_t n = 256;
    basalt::MemoryArena tight(2 * n * sizeof(float));
    EXPECT_THROW(basalt::kernel::ComplexSoA(tight, n), std::bad_alloc);

    basalt::MemoryArena tiny(64);
    EXPECT_THROW(basalt::kernel::ComplexSoA(tiny, n), std::bad_alloc);

    basalt::MemoryArena huge(4096);
    EXPECT_THROW(basalt::kernel::ComplexSoA(huge, (size_t{1} << 62) + 1), std::bad_alloc);
}

TEST(ComplexSoATest, RequiredArenaBytesSizesArenaExactly) {
    constexpr size_t n = 256;
    const size_t need = basalt::kernel::ComplexSoA::required_arena_bytes(n);
    EXPECT_GE(need, 2 * n * sizeof(float) + 2 * basalt::MemoryArena::kAliasPadding);

    basalt::MemoryArena arena(need);
    basalt::kernel::ComplexSoA soa(arena, n);
    ASSERT_NE(soa.real, nullptr);
    ASSERT_NE(soa.imag, nullptr);
    EXPECT_EQ(soa.count, n);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(soa.real) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(soa.imag) % 64, 0u);
    EXPECT_EQ(arena.bytes_remaining(), 0u);

    basalt::MemoryArena one_short(need - 1);
    EXPECT_THROW(basalt::kernel::ComplexSoA(one_short, n), std::bad_alloc);

    EXPECT_EQ(basalt::kernel::ComplexSoA::required_arena_bytes(std::numeric_limits<size_t>::max()),
              std::numeric_limits<size_t>::max());
}

// --- Move semantics / external memory ---

namespace {
int g_dealloc_calls = 0;
void* g_last_freed = nullptr;
void counting_free(void* p) {
    ++g_dealloc_calls;
    g_last_freed = p;
    std::free(p);
}
}

TEST(ArenaTest, MoveConstructorTransfersOwnership) {
    basalt::MemoryArena a(4096);
    void* p = a.allocate(100);
    ASSERT_NE(p, nullptr);
    const size_t used = a.bytes_used();

    basalt::MemoryArena b(std::move(a));
    EXPECT_EQ(b.total_size(), 4096u);
    EXPECT_EQ(b.bytes_used(), used);
    // Moved-from arena is empty and refuses allocations without crashing.
    EXPECT_EQ(a.total_size(), 0u);
    EXPECT_EQ(a.bytes_used(), 0u);
    EXPECT_EQ(a.allocate(1), nullptr);
    // The new owner keeps allocating after the old position.
    void* q = b.allocate(100);
    ASSERT_NE(q, nullptr);
    EXPECT_GT(reinterpret_cast<uintptr_t>(q), reinterpret_cast<uintptr_t>(p));
}

TEST(ArenaTest, MoveAssignmentFreesOldBlockAndTransfers) {
    g_dealloc_calls = 0;
    void* raw1 = std::aligned_alloc(64, 1024);
    void* raw2 = std::aligned_alloc(64, 2048);
    ASSERT_NE(raw1, nullptr);
    ASSERT_NE(raw2, nullptr);
    {
        basalt::MemoryArena a(raw1, 1024, &counting_free);
        basalt::MemoryArena b(raw2, 2048, &counting_free);
        a = std::move(b);
        EXPECT_EQ(g_dealloc_calls, 1);           // a's previous block released
        EXPECT_EQ(g_last_freed, raw1);
        EXPECT_EQ(a.total_size(), 2048u);
        EXPECT_EQ(b.total_size(), 0u);
    }
    EXPECT_EQ(g_dealloc_calls, 2);               // raw2 freed exactly once by a
    EXPECT_EQ(g_last_freed, raw2);
}

TEST(ArenaTest, ExternalDeallocatorConstructor) {
    g_dealloc_calls = 0;
    void* raw = std::aligned_alloc(64, 1024);
    ASSERT_NE(raw, nullptr);
    {
        basalt::MemoryArena arena(raw, 1024, &counting_free);
        EXPECT_EQ(arena.total_size(), 1024u);
        void* p = arena.allocate(128);
        EXPECT_EQ(p, raw);
        std::memset(p, 0xAB, 128);
        EXPECT_EQ(g_dealloc_calls, 0);
    }
    EXPECT_EQ(g_dealloc_calls, 1);
    EXPECT_EQ(g_last_freed, raw);
}

TEST(ArenaTest, ExternalNullBlockThrowsAndNullDeallocatorIsAllowed) {
    EXPECT_THROW(basalt::MemoryArena(nullptr, 1024, &counting_free), std::bad_alloc);

    alignas(64) static unsigned char storage[1024];
    basalt::MemoryArena arena(storage, sizeof(storage), nullptr);  // not owned
    EXPECT_NE(arena.allocate(64), nullptr);
}

TEST(ArenaTest, ExternalUnalignedBlockStillReturnsAlignedPointers) {
    alignas(64) static unsigned char storage[2048];
    basalt::MemoryArena arena(storage + 8, 1024, nullptr);
    void* p = arena.allocate(64);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
    EXPECT_GE(reinterpret_cast<unsigned char*>(p), storage + 8);
}
