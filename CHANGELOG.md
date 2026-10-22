# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Security and memory safety

- FFT, 2D FFT and transpose entry points validate sizes in every build type and throw `std::invalid_argument` for non-power-of-two sizes, null or aliased buffers and overflowing dimensions (previously a heap overflow in Release builds).
- `MemoryArena` size arithmetic is overflow-safe; huge requests return `nullptr` instead of a tiny block, and huge arena sizes throw `std::bad_alloc`.
- `ComplexSoA` and `fft2d_*` throw `std::bad_alloc` when scratch or data arenas are exhausted instead of dereferencing `nullptr`.
- SEG-Y parsing takes a buffer length everywhere and throws `basalt::io::SegyFormatError` on truncated or inconsistent input; a new `describe_segy` validates the whole file layout with overflow-checked arithmetic.
- `TaskQueue::try_push` no longer consumes the task when the queue is full (previously a full queue enqueued an empty task and aborted the process).
- NUMA `mmap` blocks are freed correctly (previously every block leaked).
- `BatchExecutor` no longer keeps a thread-local pointer to another executor's buffers (use-after-free with two executors on one thread).
- NUMA node ids and dense indices are kept separate, so sparse or more than ten nodes no longer index out of bounds.

### Added

- Variable-length SEG-Y traces: `describe_segy` switches to a per-trace index (`SegyLayout::variable_length`, `trace_offsets`, `trace_sample_counts`, `trace_samples(i)`, `trace_offset(i)`, `min_samples_per_trace`) when the binary header's fixed-length flag is 0 (revision >= 1) and the trace headers' sample counts (bytes 115-116) contradict the fixed layout. Fixed-length files that merely leave the flag at 0 keep the fast fixed layout. The walk is overflow and bounds checked; truncated files and per-trace counts pointing past the end of the file throw `SegyFormatError`.
- `SegyWriter` and `write_segy_like` preserve variable-length layouts (each trace is written with its own sample count).
- `basalt inspect` prints `Variable-length traces: yes/no` and the minimum and maximum samples per trace; `basalt filter` zero-pads variable-length traces to the longest one for the 2D FFT and crops each trace back to its own length on output.

### Fixed

- f-k filter honours `mute_vel_min`: the mute band is `[mute_vel_min, mute_vel_max]` in signed apparent velocity `v = f/k`.
- f-k filter uses the exact wavenumber (the `k + 1e-6` term is gone), keeps `H(f,k) == H(-f,-k)` exactly, and passes `k == 0` (including DC).
- f-k filter validates `dt`, `dx`, `taper_width`, velocities and buffer sizes and throws `std::invalid_argument` on bad input.
- `fast_exp_avx` clamps its input; NaN propagates through the sigmoid.
- IBM float conversion handles subnormals, signed zero, infinities and NaN, and rounds to nearest.
- EBCDIC decoding uses the real CP037 table; ASCII text headers are detected and passed through.
- `is_zero_trace` no longer treats NaN traces as dead.
- SEG-Y rev 1/2 fields are parsed: revision, fixed-length flag, extended textual headers (including the variable count), extended sample count, byte-order marker (little-endian files), trailer stanzas and all trace header fields in bytes 181-240.
- `swap_endian` is a no-op on big-endian hosts.
- `FileMapping` opens with `O_CLOEXEC`, closes the descriptor right after mapping, rejects directories and reports the OS error text.
- Exceptions thrown by thread-pool tasks or worker initializers are propagated to the caller instead of terminating the process.
- `NumaTopology::detect()` never throws and falls back to a single-node topology.
- Library worker threads enable FTZ/DAZ; idle workers block instead of polling.
- Benchmarks restore pristine input every iteration (the FFT benchmark previously timed inf/NaN data), time only the measured work, use `steady_clock` and validate results.
- `baseline.py` implements the same tapered band mute as Basalt in single precision.

### Changed

- The library baseline is plain x86-64: AVX2/FMA and AVX-512 kernels are selected at runtime from `cpu_features()`, with scalar fallbacks for the FFT, transpose and f-k filter, so Basalt no longer crashes with an illegal instruction on CPUs without AVX2. `BASALT_SIMD_MAX=scalar|avx2|avx512` caps the path; `BASALT_NATIVE_ARCH=ON` opts back into `-march=native`. The AVX-512 filter now requires only AVX-512F and DQ.
- `fft2d_forward` and `fft2d_inverse` take their scratch inside an `ArenaScope`, so caller allocations in the same arena survive and repeated calls need no `reset()`.
- Auto 2D FFT selection picks the new `RowFirstColumnStrips` schedule for every shape. Results differ from the previous Auto choice only at float rounding level.
- The `basalt` executable is a real command-line tool with `info`, `inspect` and `filter` subcommands (previously a diagnostics stub that exited 0 for any argument).
- SEG-Y parser functions take `(pointer, size)`; `SegyBinaryHeader` and `SegyTraceHeader` expose named rev 1/2 fields instead of raw padding.
- CMake no longer forces `-O3` or `-std`; Debug builds are real debug builds and consumers inherit `cxx_std_17`.
- New options `BASALT_BUILD_TESTS`, `BASALT_BUILD_BENCHMARKS`, `BASALT_BUILD_CLI` and `BASALT_INSTALL`; tests and benchmarks default on only for top-level builds.
- libnuma is no longer linked (the raw `mbind` syscall is used), so the installed package is relocatable.
- Installation uses `GNUInstallDirs`; `Basalt::basalt_core` is available as an alias for `add_subdirectory` consumers.
- One CI workflow covering GCC Release, ASan/UBSan, TSan, Clang, Windows MSVC, scalar-kernel tests, a SIMD isolation check, an install-and-consume check and clang-format; actions are pinned by commit SHA.
- The whole tree is formatted with clang-format 18 (see `.git-blame-ignore-revs`).
- `basalt_bench` exits 2 on usage errors and 0 for `--help`.

### Performance

- FFT twiddle and bit-reversal plans are precomputed and cached; the first three radix-2 stages are vectorised; the 2D inverse applies a single combined scale. 1D n=1024 AVX forward went from about 10.6 us to 2.2 us and 2D 1024x1024 forward from about 27 ms to 7.8 ms on the development machine.
- New transpose-free 2D FFT column pass (`FFT2DSchedule::RowFirstColumnStrips`) that transforms 64-column strips with SIMD lanes across columns: 1.4x to 2.0x faster than the best transpose-based schedule from 256x256 to 4096x1024.

### Added

- `fft2d_scratch_bytes`, `MemoryArena::required_bytes`, `ComplexSoA::required_arena_bytes`.
- `MemoryArena::mark()`, `MemoryArena::rewind()` and the `ArenaScope` RAII helper.
- `basalt/kernel/cpu_features.hpp` with `cpu_features()`, and `SimdMode::Scalar`.
- `describe_segy`, `read_trace_header` and `read_trace_samples` for validated trace access (IBM, IEEE and integer formats, big- and little-endian), with rev 2 trace-count and first-trace-offset checks.
- `SegyWriter` / `write_segy_like` for atomic SEG-Y output that preserves the source headers.
- `tools/check_simd_isolation.py`, which verifies that wide-SIMD instructions only appear in runtime-dispatched functions.
- Reference-DFT, sign-convention, Hermitian-symmetry, fuzz-style SEG-Y, queue back-pressure, exception propagation and fake-sysfs NUMA tests.
- `CITATION.cff`, `CONTRIBUTING.md`, `.editorconfig`, issue and pull request templates, Dependabot for GitHub Actions.

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

[Unreleased]: https://github.com/AmanSinghNp/Basalt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AmanSinghNp/Basalt/releases/tag/v0.1.0
