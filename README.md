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

## Build Instructions
```bash
mkdir build && cd build
# Build with optimizations and architecture tuning
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Project Roadmap
- [ ] Phase 1: Bare Metal I/O & SEG-Y Parsing
- [ ] Phase 2: Cache-Oblivious Matrix Transpose & 1D FFT
- [ ] Phase 3: AVX2 Vectorization & Micro-Optimization
- [ ] Phase 4: f-k Domain Logic & Sigmoid Tapering
- [ ] Phase 5: Verification & Synthetic Benchmarking

## License
MIT License
