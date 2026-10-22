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

With CMake presets (CMake 3.21 or newer for `--preset`):

```bash
cmake --preset gcc-release
cmake --build --preset gcc-release
```

Available configure presets are `gcc-release`, `gcc-debug`, `gcc-asan`, `gcc-tsan`, `clang-release`, `clang-debug` and `msvc` (build with `msvc-debug` or `msvc-release`). Build output goes to `build/<preset>/`.

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

## Command-line tool

The `basalt` executable (built when `BASALT_BUILD_CLI` is ON) has three subcommands:

- `basalt info`: version, runtime CPU features and FTZ/DAZ state.
- `basalt inspect <file.segy>`: revision, sample format, byte order, samples per trace, sample interval, trace and extended header counts, and the inline/crossline/CDP of the first and last trace.
- `basalt filter <in.segy> <out.segy> --dx M [--dt S] [--vmin V] [--vmax V] [--taper W]`: apply an f-k velocity mute to the whole file treated as one gather (time samples along one axis, traces along the other; both zero-padded to powers of two). Apparent velocities `v = f/k` between `--vmin` and `--vmax` (m/s) are muted, so the defaults (-1500 to 1500, or `-vmax..vmax` when only `--vmax` is given) remove slow events such as ground roll. `--dx` (trace spacing in meters) is required; `--dt` defaults to the binary header sample interval. The output keeps all headers byte for byte and stores samples as IEEE float, or IBM float if the input was IBM; other input sample formats are converted to IEEE float and the format code is updated. Inputs that need more working memory than 75% of RAM (about 16 bytes per padded sample) are refused; override with `--max-memory MB`.

```bash
basalt inspect shot.segy
basalt filter shot.segy shot_muted.segy --dx 12.5 --vmax 1500 --taper 200
```

`basalt --help` and `basalt --version` print usage and version. Exit status is 0 on success, 1 for runtime errors (unreadable or invalid SEG-Y, I/O errors, out of memory) and 2 for usage errors. The library equivalent of the output step is `basalt::io::SegyWriter` / `write_segy_like` in `basalt/io/segy_writer.hpp`, which writes atomically through a temporary file.

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
