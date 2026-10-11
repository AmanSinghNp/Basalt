# Basalt

![Build Status](https://github.com/AmanSinghNp/Basalt/actions/workflows/build.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![DOI](https://zenodo.org/badge/1147137556.svg)](https://doi.org/10.5281/zenodo.18907551)

Basalt filters seismic gathers in the frequency-wavenumber (f-k) domain. It removes or keeps the
events whose apparent velocity falls in a band you choose, which is the usual way to mute ground
roll and other linear noise. The core is a C++17 library. The `basalt` command line tool filters
SEG-Y files, and the `basalt` Python package works on NumPy arrays and reads and writes SEG-Y.

The f-k pipeline accepts gathers of any size, not only powers of two. It uses a vendored copy of
[pocketfft](third_party/pocketfft/README.basalt.md) (BSD-3-Clause) for its 2D real FFT.

## Requirements

- x86-64 CPU with AVX2 and FMA support
- CMake 3.15 or newer (3.18 or newer for `BASALT_BUILD_PYTHON`)
- Supported toolchains:
  - Windows: MSVC or MinGW
  - Linux: GCC or Clang
  - macOS: Clang on AVX2-capable x86_64 environments
- For the Python package: Python 3.9 or newer, NumPy and pybind11

## Quick start

### Command line

Build the `basalt` executable:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Print the geometry and sample format of a SEG-Y file, then filter it:

```bash
./build/basalt info in.segy
./build/basalt filter in.segy out.segy --vmin 500 --vmax 2000 --mode reject
```

This mutes events with apparent velocity between 500 and 2000 m/s and keeps the rest. The input
must be a SEG-Y file with a fixed trace length and IBM32, INT32, INT16, IEEE32 or INT8 samples;
other format codes are rejected. The [Python example](#python) below writes a small synthetic file,
`synthetic.segy`, which you can pass in place of `in.segy`.

Every option of `basalt filter`:

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--vmin V` | `-1500` | Lower edge of the velocity band, m/s |
| `--vmax V` | `1500` | Upper edge of the velocity band, m/s |
| `--taper T` | `0.1` | Fractional edge width, `0 <= T < 1` |
| `--mode M` | `reject` | `reject` mutes the band and keeps the rest; `pass` keeps only the band |
| `--dx M` | `25` | Trace spacing, m |
| `--dt S` | from file | Sample interval, s. A whole number of microseconds in `[1e-6, 0.065535]`. The file's binary header is used unless this is given |
| `--pad P` | `1.5` | FFT padding factor, `>= 1` |
| `--edge-taper N` | `4` | Cosine taper over the first and last `N` traces of each block. In a block shorter than `2N` traces the two tapers overlap, their weights multiply, and a warning is printed |
| `--threads N` | `1` | FFT threads. `0` uses all cores |
| `--gather-size N` | whole file | Filter consecutive blocks of `N` traces instead of the whole file as one gather |
| `--format F` | see note | Output sample format, `ibm` or `ieee`. Defaults to the input format when it is IBM32 or IEEE32, otherwise `ieee` |
| `--verbose` | off | Print SIMD and FTZ/DAZ diagnostics to stderr |

`basalt info` also accepts `--verbose`. `basalt --help` prints the summary and `basalt --version`
prints the version. Exit status is `0` on success, `1` on a runtime error (the message goes to
stderr), and `2` on a usage error.

### Python

Build the extension, then point `PYTHONPATH` at the build tree:

```bash
cmake -S . -B build -DBASALT_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Or install the package into the active environment with `pip install .`, which builds the
extension with scikit-build-core.

This example builds a synthetic gather with two linear events, filters it, and writes and reads
a SEG-Y file. Save it as `example.py`:

```python
import numpy as np
import basalt

dt, dx = 0.002, 5.0                      # 2 ms samples, 5 m trace spacing
ntr, ns = 96, 600
t = np.arange(ns) * dt
x = np.arange(ntr) * dx


def ricker(tau, f=25.0):
    a = (np.pi * f * tau) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


# Two linear events: +1000 m/s (arrival time rises with offset) and -1000 m/s (it falls).
gather = np.zeros((ntr, ns))
for t0, v in ((0.2, 1000.0), (0.8, -1000.0)):
    for i in range(ntr):
        gather[i] += ricker(t - (t0 + x[i] / v))
gather = gather.astype(np.float32)

# Reject the band [500, 2000] m/s: the +1000 m/s event is strongly attenuated, and the
# -1000 m/s event is largely kept.
filtered = basalt.fk_filter(gather, dt, dx, v_min=500.0, v_max=2000.0, mode="reject")
print(filtered.shape, filtered.dtype)

# Mute weight of one f-k bin at 25 Hz and k = -0.025 cycles/m, i.e. v = -f/k = +1000 m/s:
# inside the band, so the weight is 0.
print(basalt.fk_mute_weight(25.0, -0.025, v_min=500.0, v_max=2000.0))

# SEG-Y round trip.
basalt.write_segy("synthetic.segy", filtered, dt, format="ieee")
d = basalt.read_segy("synthetic.segy")
print(d["samples"].shape, d["dt"], d["format"])
```

Run it with the build tree on the path (after `pip install .`, leave out `PYTHONPATH`):

```bash
PYTHONPATH=build/python python3 example.py
```

`fk_filter` takes a 2D array of shape `(ntraces, nsamples)`, trace-major, and returns a new
float32 array of the same shape. Other dtypes and non-contiguous arrays are converted. The
signature is in `python/basalt/__init__.py` and the contract is in
[docs/design/FK_PIPELINE.md](docs/design/FK_PIPELINE.md), section 7.

### C++

Install the library, headers and CMake package:

```bash
cmake --install build --prefix "$HOME/.local"
```

In your project, find the package and link the `Basalt::basalt_core` target:

```cmake
cmake_minimum_required(VERSION 3.15)
project(demo CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Basalt 0.2 REQUIRED)

add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE Basalt::basalt_core)
```

If the package is under a non-default prefix, pass it with `-DCMAKE_PREFIX_PATH=<prefix>` when you
configure the project. `demo.cpp` filters a gather held in a `std::vector<float>`:

```cpp
#include <basalt/fk.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const size_t ntr = 96, ns = 600;  // trace-major: gather[t * ns + s]
    const double dt = 0.002, dx = 5.0;
    const double pi = 3.14159265358979323846;

    // Two linear events: +1000 m/s (arrival time rises with offset) and -1000 m/s.
    const struct { double t0, v; } events[] = {{0.2, 1000.0}, {0.8, -1000.0}};
    std::vector<float> gather(ntr * ns, 0.0f);
    for (const auto& e : events) {
        for (size_t x = 0; x < ntr; ++x) {
            const double t_arrival = e.t0 + (static_cast<double>(x) * dx) / e.v;
            for (size_t s = 0; s < ns; ++s) {
                const double a = pi * 25.0 * (static_cast<double>(s) * dt - t_arrival);
                gather[x * ns + s] += static_cast<float>((1.0 - 2.0 * a * a) * std::exp(-a * a));
            }
        }
    }

    basalt::FKMuteParams params;
    params.dt = dt;
    params.dx = dx;
    params.v_min = 500.0;
    params.v_max = 2000.0;
    params.mode = basalt::FKMode::Reject;  // mute the band, keep everything else

    const std::vector<float> filtered = basalt::fk_filter_gather(gather, ntr, ns, params);

    double e_in = 0.0, e_out = 0.0;
    for (size_t i = 0; i < gather.size(); ++i) {
        e_in += static_cast<double>(gather[i]) * gather[i];
        e_out += static_cast<double>(filtered[i]) * filtered[i];
    }
    std::printf("energy after reject: %.3f of input\n", e_out / e_in);
    return 0;
}
```

The header `basalt/fk.hpp` declares `basalt::fk_filter_gather` (pointer and vector overloads),
`basalt::FKMuteParams` and `basalt::FKMode`. The pipeline is specified in
[docs/design/FK_PIPELINE.md](docs/design/FK_PIPELINE.md), section 4. `basalt::fk_mute_weight`
returns the weight of any single f-k bin for the same parameters.

## Apparent velocity and the mute modes

Each f-k sample has apparent velocity `v = -f / k`. An event whose arrival time rises with trace
position (`t = t0 + x / V` with `V > 0`) has `v = +V`, and one whose time falls has `v = -V`.
`FKMode::Reject` mutes the samples whose velocity lies in `[v_min, v_max]` and keeps the rest,
and `FKMode::Pass` keeps only that band. The band edges are raised-cosine tapers with half-width
`0.5 * taper * |edge|`. The full definition, including the `k = 0` and DC cases, is in
[docs/design/FK_PIPELINE.md](docs/design/FK_PIPELINE.md), sections 2 and 3.

## Performance

In the measurements in [PERFORMANCE.md](PERFORMANCE.md), the f-k pipeline is faster than numpy's
2D FFT round trip at every size and thread count tested. Single-threaded, it is roughly on par
with `scipy.fft`, sometimes a little ahead and sometimes a little behind. With four threads
`scipy.fft` was ahead in those runs. The numbers come from one 4-vCPU cloud VM and are indicative,
not certified. Methodology, hardware and caveats are in [PERFORMANCE.md](PERFORMANCE.md).

## Legacy power-of-two API

The original API works on single-precision complex data in a structure-of-arrays layout
(`ComplexSoA`), and it requires both dimensions to be powers of two. It is declared in
`basalt/filter/fk_filter.hpp` and `basalt/kernel/fft2d.hpp`. Its velocity is `v = f / k`, the
opposite sign to the pipeline above. Use the pipeline for new code.

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

## Build and install

The C++ library, the `basalt` executable, the tests and the benchmarks build with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Add `-DBASALT_BUILD_PYTHON=ON` to build the Python extension as well. Install the C++ parts with:

```bash
cmake --install build --prefix <install-prefix>
```

## Testing

```bash
cd build
ctest --output-on-failure
```

With `-DBASALT_BUILD_PYTHON=ON`, `ctest` also runs the Python tests, which need NumPy and pybind11.
Continuous integration builds Release and a sanitizer (AddressSanitizer and UBSan, Debug) configuration
on Linux, and has a separate job for the Python bindings. See
[.github/workflows/ci.yml](.github/workflows/ci.yml).

## Benchmark tools

- `basalt_bench` runs end-to-end matrix sweeps over size, threads, SIMD mode, NUMA mode and memory policy
- `fk_gather_bench` times the f-k pipeline (`basalt::fk_filter_gather`) on full gathers
- `fk_benchmark` times the legacy f-k filter kernel
- `fft_benchmark` times the 1D FFT kernels
- `fft2d_schedule_benchmark` compares 2D FFT scheduling strategies

`basalt_bench` example:

```bash
./build/basalt_bench --sizes 256,512,1024 --threads 1,2,4 --simd auto,avx2,avx512 --numa off,auto --memory-policy bind-worker-buffers,bind-inputs-if-possible --csv-out benchmark.csv --json-out benchmark.json
```

CSV output uses flat `|`-delimited counter fields for locality metrics. JSON emits the same
metrics as numeric arrays. The comparison against NumPy and SciPy, and how to reproduce it, are
in [PERFORMANCE.md](PERFORMANCE.md).

## Project layout

- `include/basalt/` public headers (`fk.hpp`, `io/segy_file.hpp`, and the legacy kernel headers)
- `src/` library implementation, including `src/fk/` (f-k pipeline) and `src/io/` (SEG-Y)
- `src/main.cpp` the `basalt` command line tool
- `python/` pybind11 bindings (`python/src`), the Python package (`python/basalt`) and its tests
- `benchmarks/` performance tools and the NumPy/SciPy comparison
- `tests/` integrity and integration coverage
- `docs/design/FK_PIPELINE.md` the design contract for the pipeline, SEG-Y I/O, CLI and Python API
- `third_party/pocketfft/` vendored pocketfft (BSD-3), see its `README.basalt.md`
- `tools/` profiling and workflow helpers
- `PERFORMANCE.md` benchmark methodology and results

## Documentation

- [docs/design/FK_PIPELINE.md](docs/design/FK_PIPELINE.md): the f-k pipeline, SEG-Y, CLI and Python contract
- [PERFORMANCE.md](PERFORMANCE.md): benchmark methodology and reporting notes
- [CHANGELOG.md](CHANGELOG.md): release history

## License

MIT License. See [LICENSE](LICENSE). The vendored pocketfft is BSD-3-Clause; see
[third_party/pocketfft/README.basalt.md](third_party/pocketfft/README.basalt.md).
