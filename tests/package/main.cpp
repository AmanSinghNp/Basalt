#include <basalt/arena.hpp>

#include <cstdio>

int main() {
    basalt::MemoryArena arena(4096);
    void* p = arena.allocate(256);
    if (p == nullptr || arena.bytes_used() < 256) {
        std::fprintf(stderr, "arena allocation failed\n");
        return 1;
    }
    std::puts("basalt package consumer ok");
    return 0;
}
