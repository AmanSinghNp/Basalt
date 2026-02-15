#include <iostream>
#include <immintrin.h>
#include "basalt/kernel/cpu_state.hpp"

// Simple architecture check
void print_cpu_capabilities() {
    std::cout << "[Basalt] System Architecture Check:\n";
    
    #ifdef __AVX2__
        std::cout << "  [OK] AVX2 Supported\n";
    #else
        std::cerr << "  [FAIL] AVX2 Not Detected! Performance will be critical.\n";
    #endif

    #ifdef __FMA__
        std::cout << "  [OK] FMA Supported\n";
    #else
        std::cerr << "  [WARN] FMA Not Detected.\n";
    #endif
}

int main(int argc, char* argv[]) {
    // Critical: set FTZ/DAZ before any floating-point computation
    basalt::kernel::set_flush_to_zero();

    std::cout << "Basalt: Seismic Processing Engine v0.2\n";
    std::cout << "======================================\n";
    
    print_cpu_capabilities();

    if (basalt::kernel::check_ftz_daz()) {
        std::cout << "  [OK] FTZ/DAZ Enabled (denormals flushed to zero)\n";
    } else {
        std::cerr << "  [WARN] FTZ/DAZ flags not set!\n";
    }

    if (argc < 2) {
        std::cout << "\nUsage: ./basalt <input.segy>\n";
        return 1;
    }

    return 0;
}
