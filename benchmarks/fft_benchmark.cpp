#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include "basalt/kernel/fft_avx.hpp"
#include "basalt/kernel/fft2d.hpp"
#include "basalt/kernel/complex_soa.hpp"

using namespace basalt::kernel;

void benchmark_fft(size_t n) {
    basalt::MemoryArena arena(n * sizeof(float) * 4);
    ComplexSoA data(arena, n);
    
    // Setup data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data.real[i] = dist(rng);
        data.imag[i] = dist(rng);
    }
    
    // Warmup
    for(int i=0; i<100; ++i) {
        fft_forward_avx(data.real, data.imag, n);
    }
    
    // Measure
    int iterations = 10000;
    if (n > 4096) iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        fft_forward_avx(data.real, data.imag, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_sec = elapsed.count() / iterations;
    double throughput_mitems = (static_cast<double>(n) * iterations) / elapsed.count() / 1e6;
    
    std::cout << "FFT Size " << n << ": " 
              << avg_time_sec * 1e6 << " us/op, " 
              << throughput_mitems << " MItems/s" << std::endl;
}

void benchmark_fft2d(size_t rows, size_t cols) {
    std::cout << "Initializing 2D FFT " << rows << "x" << cols << "..." << std::endl;
    basalt::MemoryArena arena(rows * cols * sizeof(float) * 4);
    ComplexSoA data(arena, rows * cols);
    
    for(size_t i=0; i<rows*cols; ++i) { data.real[i] = 1.0f; data.imag[i] = 0.0f; }
    
    basalt::MemoryArena scratch(rows * cols * sizeof(float) * 2 + 4096);
    
    std::cout << "Calling fft2d_forward..." << std::endl;
    fft2d_forward(data.real, data.imag, rows, cols, scratch);
    std::cout << "fft2d_forward done." << std::endl;
}

int main() {
    std::cout << "Benchmarking AVX FFT..." << std::endl;
    std::vector<size_t> sizes = {256, 512, 1024};
    for(size_t n : sizes) {
        benchmark_fft(n);
    }
    
    std::cout << "Benchmarking 2D FFT..." << std::endl;
    benchmark_fft2d(256, 256);
    
    return 0;
}
