# Basalt
**High-Performance Seismic f-k Muter**

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

## Project Roadmap
- [x] Phase 1: Bare Metal I/O & SEG-Y Parsing
- [x] Phase 2: Cache-Oblivious Matrix Transpose & 1D FFT
- [x] Phase 3: AVX2 Vectorization & Micro-Optimization
- [x] Phase 4: f-k Domain Logic & Sigmoid Tapering
- [x] Phase 5: Verification & Synthetic Benchmarking
- [x] Phase 6: Release Optimization & Final Polish

## License
MIT License
