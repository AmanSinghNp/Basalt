# Basalt
**High-performance seismic f-k muting library for x86_64**

![Build Status](https://github.com/AmanSinghNp/Basalt/actions/workflows/build.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Basalt is a C++ library for seismic f-k filtering with an architecture-aware implementation: tiled FFT scheduling, SIMD-dispatched filtering, aligned arena allocation, and an internal throughput runtime for multi-shot benchmarking.

The primary product is the `basalt_core` library. The `basalt` executable is a small diagnostics stub, and the benchmark binaries are the main operational entrypoints for performance work.

## Current Status
- Cache-aware 2D FFT scheduling is implemented with locked auto-tuning defaults.
- The f-k filter supports `auto`, `avx2`, and `avx512` SIMD selection.
- The AVX-512 path currently applies only to the f-k filter, not the FFT.
- Internal batch execution, thread-pool scheduling, and NUMA-aware worker placement exist for throughput benchmarking.
- PMU-backed certification still requires representative self-hosted Linux hardware.

Detailed benchmark/reporting guidance lives in [PERFORMANCE.md](./PERFORMANCE.md).

## Architecture
Basalt’s current design is split into a few focused layers:

1. **I/O and parsing**: file mapping and SEG-Y parsing.
2. **Memory**: 64-byte aligned arena allocation with reusable scratch storage.
3. **Kernel layer**: FFT, tiled transpose, and SIMD-aware filtering.
4. **Internal runtime**: task queue, thread pool, NUMA topology detection, and batch execution for benchmark sweeps.
5. **Benchmark/reporting tools**: latency/throughput sweeps and schedule comparison.

`include/basalt/internal` contains implementation internals and is not part of the supported public API.

## Requirements
- OS: Windows (MSVC), Linux (GCC/Clang), macOS on AVX2-capable x86_64 environments
- Compiler: C++17
- Build system: CMake 3.15+
- Hardware baseline: x86_64 CPU with AVX2 and FMA

### AVX-512 support
- GCC/Clang builds can use the optimized AVX-512 f-k filter path when the CPU supports the required feature set.
- MSVC and non-supporting builds safely fall back to AVX2 behavior when `SimdMode::AVX512` is requested.
- FFT kernels remain AVX2-only in this pass.

## Build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Install
Basalt installs the `basalt_core` library, public headers, and CMake package metadata for downstream consumers.

```bash
cmake --install . --config Release --prefix <install-prefix>
```

## Tests
```bash
ctest --output-on-failure
```

## Benchmarks
Available benchmark binaries:
- `basalt_bench`: end-to-end sweep across size, thread count, SIMD mode, NUMA mode, and memory policy
- `fft2d_schedule_benchmark`: compare FFT schedule/tile choices
- `fft_benchmark`: focused 1D FFT checks
- `fk_benchmark`: focused filter benchmark

Example:

```bash
./basalt_bench --sizes 256,512,1024 --threads 1,2,4 --simd auto,avx2,avx512 --numa off,auto --memory-policy bind-worker-buffers,bind-inputs-if-possible --csv-out benchmark.csv --json-out benchmark.json
```

`basalt_bench` emits:
- CSV with flat per-row counters encoded as `|`-delimited lists
- JSON with the same counters emitted as numeric arrays

## Library Usage
```cpp
#include <basalt/arena.hpp>
#include <basalt/filter/fk_filter.hpp>
#include <basalt/kernel/complex_soa.hpp>
#include <basalt/kernel/fft2d.hpp>

size_t rows = 1024;
size_t cols = 1024;
size_t total = rows * cols;

basalt::MemoryArena arena(total * sizeof(float) * 2 + 4096);
basalt::kernel::ComplexSoA data(arena, total);

basalt::MemoryArena scratch(total * sizeof(float) * 2 + 8192);
basalt::kernel::FFT2DConfig fft_config;

basalt::kernel::fft2d_forward(data.real, data.imag, rows, cols, scratch, fft_config);

basalt::filter::FKParams params;
params.mute_vel_min = -1500.0;
params.mute_vel_max = 1500.0;
params.taper_width = 100.0;
basalt::filter::apply_fk_filter(data, rows, cols, params);

scratch.reset();
basalt::kernel::fft2d_inverse(data.real, data.imag, rows, cols, scratch, fft_config);
```

## CLI
`basalt` is currently a diagnostics/demo executable. It reports CPU state and build assumptions; it is not yet a production processing CLI.

## Documentation
- [PERFORMANCE.md](./PERFORMANCE.md): benchmark methodology, reporting fields, and current optimization status
- `optimisation*.md`: engineering planning/reference material

## License
MIT License
