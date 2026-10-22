# Basalt

High-performance seismic f-k muting library for x86_64 systems.

![Build Status](https://github.com/AmanSinghNp/Basalt/actions/workflows/build.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![DOI](https://zenodo.org/badge/1147137556.svg)](https://doi.org/10.5281/zenodo.18907551)


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
- any x86_64 CPU: Basalt detects AVX2/FMA and AVX-512 at runtime and uses them when available, with portable scalar fallbacks otherwise. Set `BASALT_SIMD_MAX=scalar|avx2|avx512` to cap the instruction set (see `basalt/kernel/cpu_features.hpp`); configure with `-DBASALT_NATIVE_ARCH=ON` to build for the build machine's CPU (`-march=native`)

Tested in CI: Linux with GCC and Clang. Windows (MSVC, MinGW) and macOS (x86_64 Clang) are expected to work but are not currently tested in CI.

## Build

With CMake presets (CMake 3.19 or newer for `--preset`):

```bash
cmake --preset gcc-release
cmake --build --preset gcc-release
```

Available presets are `gcc-release`, `gcc-debug`, `clang-debug` and `msvc-debug`. Build output goes to `build/<preset>/`.

Or configure manually:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

Build options:

- `BASALT_BUILD_TESTS`: build the test suite (ON by default in a top-level build)
- `BASALT_BUILD_BENCHMARKS`: build the benchmark executables (ON by default in a top-level build)
- `BASALT_BUILD_CLI`: build the `basalt` diagnostic CLI (ON by default in a top-level build)

## Install

Basalt installs the `basalt_core` library, public headers, and CMake package metadata. When `BASALT_BUILD_CLI` is ON, the install also includes the `basalt` diagnostic CLI.

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

Notes:

- `rows` and `cols` must be powers of two. The FFT and filter functions throw `std::invalid_argument` on invalid sizes.
- The mute band is `[mute_vel_min, mute_vel_max]`, measured in signed apparent velocity `v = f / k`. Bins with `k == 0` (infinite apparent velocity) are passed through unmuted.

## Testing

```bash
ctest --output-on-failure
```

## Citation

If you use Basalt in published work, please cite it. Machine-readable citation metadata is in [CITATION.cff](./CITATION.cff). DOI: [10.5281/zenodo.18907551](https://doi.org/10.5281/zenodo.18907551).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the build and test workflow, formatting, commit conventions and benchmarks.

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
