#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "basalt/filter/fk_filter.hpp"
#include "basalt/kernel/complex_soa.hpp"

using namespace basalt::filter;
using namespace basalt::kernel;

void benchmark_fk(size_t rows, size_t cols) {
    basalt::MemoryArena arena(rows * cols * sizeof(float) * 2 + 4096);
    ComplexSoA data(arena, rows * cols);
    
    std::fill_n(data.real, rows * cols, 1.0f);
    std::fill_n(data.imag, rows * cols, 0.0f);
    
    FKParams params;
    
    // Warmup
    for(int i=0; i<10; ++i) apply_fk_filter(data, rows, cols, params);
    
    // Measure
    int iterations = 1000;
    if (rows * cols > 1024*1024) iterations = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        apply_fk_filter(data, rows, cols, params);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_sec = elapsed.count() / iterations;
    double cells = static_cast<double>(rows) * static_cast<double>(cols);
    double throughput_mcells = (cells * iterations) / elapsed.count() / 1e6;
    double throughput_gcells = throughput_mcells / 1000.0;
    
    std::cout << "FK Grid " << rows << "x" << cols << " (" << cells/1e6 << " MCells): " 
              << avg_time_sec * 1000.0 << " ms/op, " 
              << throughput_gcells << " GCells/s" << std::endl;
}

int main() {
    std::cout << "Benchmarking F-K Filter Kernel..." << std::endl;
    benchmark_fk(256, 256);
    benchmark_fk(512, 512);
    benchmark_fk(1024, 1024);
    benchmark_fk(2048, 2048);
    return 0;
}
