# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- `basalt.read_trace_headers(path)` returns the per-trace header fields as a dict of 1D NumPy arrays, one entry per trace, with the dtypes listed in `docs/design/FK_PIPELINE.md`.
  The SEG-Y scalar rule is applied to coordinates and elevations. Only the file and trace headers are read,
  so the samples are never decoded.
- `basalt.read_segy` result gains a `headers` key holding the same dict. Existing keys are unchanged.
- C++ `basalt/io/trace_headers.hpp` with `TraceHeaders`, `decode_trace_headers` and `read_trace_headers`, and
  `read_segy_headers` in `basalt/io/segy_file.hpp`, which reads the same layout as `read_segy` without samples
- Tests for the trace header decoding in C++ and Python: every key, positive, negative and zero scalars, and
  `coord_units` values
- `basalt.write_segy(path, samples, dt, format="ieee", headers=None)` takes an optional dict of per-trace header
  arrays, with the keys of `read_trace_headers`. Coordinates and elevation are stored as `round(value / scale)`,
  with scalars chosen over all traces (-100 when a coordinate or elevation is not whole in any trace, otherwise 1) unless the dict gives them.
  Unknown keys, wrong lengths, non-whole integer values and values that overflow their field raise `ValueError`,
  and no file is written.
- C++ `TraceHeaderValues` and `encode_trace_headers` in `basalt/io/trace_headers.hpp`, the inverse of
  `decode_trace_headers`
- Round-trip tests for header writing: every key, arcsecond coordinates for a station near lat 27.7 and lon 85.3,
  time fields and delay, given and default scalars, the byte positions in the file, and the error cases

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

- `fk_filter_gather` caches the per-bin weight table per (padded size, parameters), reuses per-thread
  workspaces, zeroes only the padding, and splits the copy-in, weight multiply and copy-out across
  `threads`. The end-to-end call is now within 3 to 11% of the bare FFT pair single-threaded.

### Changed

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
