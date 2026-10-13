# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.2.0] - 2026-10-11

### Added

- End-to-end f-k mute pipeline, `basalt::fk_filter_gather` in `basalt/fk.hpp`: pocketfft real-to-complex
  2D FFT, padding to good FFT sizes (any gather size works, including non power-of-two sizes), cosine
  edge tapers on the traces and samples, signed apparent-velocity bands with raised-cosine edges, and
  `Reject` and `Pass` modes. `basalt::fk_mute_weight` exposes the per-bin weight.
- Design contract `docs/design/FK_PIPELINE.md` covering the f-k pipeline, SEG-Y I/O, the CLI and the Python API
- SEG-Y reader and writer in `basalt/io/segy_file.hpp`. Reads IBM32, INT32, INT16, IEEE32 and INT8 samples
  and writes IBM32 and IEEE32. Headers are kept as raw bytes and written back from them. On write, the binary
  header's format code is set to the output format and its extended textual header count is set to 0.
- `basalt` command line tool with `info` and `filter` commands. `filter` runs on whole files or in blocks of
  traces (`--gather-size`), selects the output sample format, and returns exit status 0, 1 or 2 for success,
  runtime errors and usage errors. `filter --dt` rewrites the sample-interval fields in the binary header and in
  each trace header that has one.
- Python package `basalt` built with pybind11 (`-DBASALT_BUILD_PYTHON=ON`, or `pip install .`), providing
  `fk_filter`, `fk_mute_weight`, `read_segy` and `write_segy`
- Benchmarks of the f-k pipeline: `fk_gather_bench` (C++, with legacy and pocketfft-only rows) and
  `benchmarks/compare_numpy.py` (against numpy and scipy). Results and methodology are in `PERFORMANCE.md`.
- Vendored pocketfft (BSD-3-Clause, header-only) in `third_party/pocketfft`, with a provenance note in `README.basalt.md`
- Test coverage for the pipeline, SEG-Y I/O, the CLI and the Python bindings
- CI job that builds and tests the Python bindings

### Performance

- Staged f-k transform: the time-axis FFT skips the zero-padded traces and the traces that are
  cropped away, and frequency columns whose mute weights are all 1 (or all 0) skip the spatial FFT
  pair. With the default ground-roll mute this is 1.7x to 2.9x faster single-threaded than the
  first 0.2.0 pipeline; with a band that touches every column it is about 1.35x faster.
- `fk_filter_gathers` batch API: filters many equal-size gathers in parallel on a pool of workers
  that sleep when idle and exit after 2 s without work. Higher throughput than a loop of
  `fk_filter_gather` calls for 64 gathers of 240x1001 at 4 threads (3.1x).
- `basalt filter` uses the batch API and threads the SEG-Y codec: 1.8x faster at `--threads 1`
  and 2.7x at `--threads 4` on a 512 MB IBM file.
- SEG-Y reader and writer: bulk reads, vectorised IBM and IEEE conversion, and an optional
  `threads` argument on `read_segy` / `write_segy` (results do not depend on it).
- Python `fk_filter`: the input finiteness pass is gone (the C++ call checks it) and the output
  overflow check runs without the GIL.

- `fk_filter_gather` caches the per-bin weight table per (padded size, parameters), reuses per-thread
  workspaces, zeroes only the padding, and splits the copy-in, weight multiply and copy-out across
  `threads`. The end-to-end call is now within 3 to 11% of the bare FFT pair single-threaded.

### Changed

- `fk_filter_gather` and `fk_filter_gathers` throw `std::invalid_argument` when the input contains
  NaN or infinite samples, before writing any output. Earlier builds let such samples spread
  through the whole gather. The CLI reports this as a runtime error (exit code 1), and Python as
  `ValueError`.
- `basalt filter --gather-size N` output no longer depends on `--threads` when the file splits
  into two or more gathers (it is bit-identical to `--threads 1`).

- Project version is 0.2.0 (CMake project version, `basalt --version`, Python `basalt.__version__`)
- README rewritten around the pipeline, the `basalt` CLI, the Python package and the C++ package; the legacy
  power-of-two API is documented in its own section
- Legacy `apply_fk_filter` output differs from 0.1.0. The causes are the signed band and the DC and `k = 0`
  handling listed under Fixed, which change which bins are muted and how the velocity divisor is computed
- CI: the sanitizer job fails on any UBSan finding (`-fno-sanitize-recover=undefined`), and the benchmark
  steps no longer ignore failures

### Fixed

- `file_mapping_test` used one temporary file name for every test, so tests run in parallel by `ctest -j`
  could delete each other's file. Each test now uses its own name
- Under AddressSanitizer, `ctest` sets `ASAN_OPTIONS=allocator_may_return_null=1`, so the CLI's out-of-memory
  path is tested instead of aborting the process

- Legacy `apply_fk_filter` ignored `mute_vel_min`, so the mute was always symmetric in `|v|`. It now mutes the
  signed band `[mute_vel_min, mute_vel_max]`
- Legacy `apply_fk_filter` muted the DC bin and so removed the data mean. The DC bin is now kept
- Legacy `apply_fk_filter` handled `k = 0` by adding `1e-6` to the divisor. Bins with `k = 0` now get weight 1
  exactly, and the divisor is no longer offset

## [0.1.0] - 2026-03-08

### Added

- Initial public release of the `basalt_core` library
- 1D and 2D FFT kernels with tiled transpose support
- Automatic 2D FFT schedule selection with benchmark coverage
- SIMD-dispatched f-k filtering with AVX2 and optional AVX-512 filter execution
- Internal task queue, thread pool, batch executor, and NUMA-aware runtime support for benchmark workflows
- End-to-end benchmark tooling with CSV and JSON reporting
- CMake install and package export support for downstream consumers
- Integrity and integration test coverage across kernel, filter, runtime, I/O, and pipeline paths
- Performance reporting document and release-oriented project documentation
