# Basalt
**High-Performance Seismic f-k Muter**

![Build Status](https://github.com/AmanSinghNp/Basalt/actions/workflows/build.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Basalt is an architecture-aware compute engine designed to process seismic data at the physical limits of x86_64 hardware. It implements a Frequency-Wavenumber (f-k) dip filter from scratch, bypassing standard libraries to optimize for L1/L2 cache latency and SIMD throughput.

**Target Performance:** >50x speedup over Python/NumPy.

## Architecture: "The Iron Foundation"
Basalt ignores high-level abstractions in favor of direct hardware control:
1.  **I/O:** `mmap`-based zero-copy loading.
2.  **Memory:** Custom Arena Allocator with 64-byte alignment (AVX2-ready).
3.  **Compute:** Hand-written AVX2/FMA intrinsics for FFT and Muting.
4.  **Math:** Branchless mask generation and polynomial Sigmoid approximation.

## Performance
- **Speedup**: **~40x** faster than Python/NumPy baseline (on 1024x1024 grid).
- **Throughput**: **~2.7 GB/s** (Compute Bound efficiency).
- **Latency**: 3ms per shot (1024x1024).

## Requirements
* **OS**: Windows (MSVC), Linux (GCC/Clang), macOS (Clang - if AVX2 available via Rosetta/native).
* **Compiler**: C++17 compliant.
* **Build System**: CMake 3.15+.
* **Hardware**: x86_64 CPU with **AVX2** and **FMA** instructions (Haswell+).

## Build Instructions
```bash
mkdir build && cd build
# Build for Release with AVX2 optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Running Tests
```bash
./basalt_tests.exe
```

### Running Benchmarks
```bash
./fft_benchmark.exe
./fk_benchmark.exe
```

## Library Usage
Basalt is primarily a high-performance library. Example usage:

```cpp
#include <basalt/arena.hpp>
#include <basalt/kernel/complex_soa.hpp>
#include <basalt/kernel/fft2d.hpp>
#include <basalt/filter/fk_filter.hpp>

// 1. Setup Data
size_t rows = 1024, cols = 1024;
basalt::MemoryArena arena(256 * 1024 * 1024);
basalt::kernel::ComplexSoA data(arena, rows * cols);
// ... load data into data.real ...

// 2. Processing Pipeline
basalt::MemoryArena scratch(rows * cols * sizeof(float) * 2 + 4096);
basalt::kernel::fft2d_forward(data.real, data.imag, rows, cols, scratch);

basalt::filter::FKParams params;
params.mute_vel_min = -1500;
params.mute_vel_max = 1500;
params.taper_width = 100;
basalt::filter::apply_fk_filter(data, rows, cols, params);

scratch.reset(); // Reuse scratch memory
basalt::kernel::fft2d_inverse(data.real, data.imag, rows, cols, scratch);
```

## Project Roadmap
- [x] Phase 1: Bare Metal I/O & SEG-Y Parsing
- [x] Phase 2: Cache-Oblivious Matrix Transpose & 1D FFT
- [x] Phase 3: AVX2 Vectorization & Micro-Optimization
- [x] Phase 4: f-k Domain Logic & Sigmoid Tapering
- [x] Phase 5: Verification & Synthetic Benchmarking
- [x] Phase 6: Release Optimization & Final Polish

## License
MIT License
