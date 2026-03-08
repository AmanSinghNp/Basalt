# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

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
