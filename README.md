# Basalt

High-performance seismic f-k muting library for x86_64 systems.

![Build Status](https://github.com/AmanSinghNp/Basalt/actions/workflows/build.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

Basalt is a C++ library for seismic frequency-wavenumber filtering with an implementation designed around cache efficiency, aligned memory access, SIMD execution, and reproducible benchmarking.

The project provides:

- a core library, `basalt_core`
- benchmark tools for FFT, filter, and end-to-end pipeline evaluation
- installation and CMake package support for downstream consumers
- correctness and regression coverage across FFT, transpose, filtering, runtime, and I/O components

## Features

- 64-byte aligned arena allocator for predictable scratch memory usage
- 1D and 2D FFT kernels with tiled transpose support
- automatic 2D FFT schedule selection with tuned defaults
- SIMD-dispatched f-k filtering with AVX2 and optional AVX-512 execution paths
- internal batch execution runtime for threaded and NUMA-aware benchmark sweeps
- CSV and JSON benchmark output for automation and reporting

## Requirements

- C++17
- CMake 3.15 or newer
- x86_64 CPU with AVX2 and FMA support
- Supported toolchains:
  - Windows: MSVC or MinGW
  - Linux: GCC or Clang
  - macOS: Clang on AVX2-capable x86_64 environments

## Build

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Install

Basalt installs the `basalt_core` library, public headers, and CMake package metadata.

```bash
cmake --install . --config Release --prefix <install-prefix>
```

## Usage

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

## Testing

```bash
ctest --output-on-failure
```

## Benchmark Tools

Basalt includes several benchmark executables:

- `basalt_bench` for end-to-end matrix sweeps across size, threads, SIMD mode, NUMA mode, and memory policy
- `fft2d_schedule_benchmark` for comparing 2D FFT scheduling strategies
- `fft_benchmark` for focused FFT benchmarking
- `fk_benchmark` for focused filter benchmarking

Example:

```bash
./basalt_bench --sizes 256,512,1024 --threads 1,2,4 --simd auto,avx2,avx512 --numa off,auto --memory-policy bind-worker-buffers,bind-inputs-if-possible --csv-out benchmark.csv --json-out benchmark.json
```

Benchmark output:

- CSV uses flat `|`-delimited counter fields for locality metrics
- JSON emits those same metrics as numeric arrays

## Project Layout

- `include/basalt/` public headers
- `src/` library implementation
- `benchmarks/` performance tools
- `tests/` integrity and integration coverage
- `tools/` profiling and workflow helpers
- `PERFORMANCE.md` benchmark methodology and reporting notes

## Documentation

- [PERFORMANCE.md](./PERFORMANCE.md)
- [CHANGELOG.md](./CHANGELOG.md)

## License

MIT License. See [LICENSE](./LICENSE).
