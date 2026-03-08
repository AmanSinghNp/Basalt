# Basalt Performance Report

## Executive Summary
Basalt now has the core scaffolding needed to validate optimization stages end to end:

- cache-aware FFT schedule selection is wired into the default path
- the f-k filter supports tiled traversal and runtime SIMD dispatch
- an internal batch executor, thread pool, NUMA topology detector, and node-local worker-buffer allocator exist for throughput sweeps
- the benchmark harness can sweep size, thread count, SIMD mode, NUMA mode, and memory policy and emit CSV/JSON

Release certification is still gated on representative self-hosted Linux hardware. GitHub-hosted runners remain unsuitable for PMU-backed performance signoff.

## Representative Hardware
Record benchmark hardware here for every published run.

| Field | Value |
| :--- | :--- |
| CPU model | `TBD` |
| Microarchitecture | `TBD` |
| Core / thread count | `TBD` |
| NUMA topology | `TBD` |
| RAM / channels | `TBD` |
| OS / kernel | `TBD` |
| Compiler | `TBD` |

## Methodology
- Use `build/Release` or equivalent optimized configuration.
- Pin CPU frequency/governor before measurements on Linux.
- Use `basalt_bench` for matrix sweeps and `fft2d_schedule_benchmark` for tile/schedule tuning.
- Keep PMU validation on self-hosted Linux with usable `perf` counters.
- Treat GitHub Actions as correctness CI, not as representative performance hardware.

### Benchmark Commands
```bash
./basalt_bench --sizes 256,512,1024,2048,4096 --threads 1,2,4,8,16 --simd auto,avx2,avx512 --numa off,auto --memory-policy bind-worker-buffers,bind-inputs-if-possible --shots 64 --warmup 8 --csv-out benchmark.csv --json-out benchmark.json
./fft2d_schedule_benchmark --size 1024 --iterations 10 --warmup 2
```

`basalt_bench` writes locality counters in two formats:
- CSV: `preferred_node_counts`, `worker_home_node_counts`, `local_queue_executions_per_node`, and `remote_steals_from_node` are `|`-delimited flat fields.
- JSON: the same fields are emitted as numeric arrays for downstream tooling.

## Results Table
Populate from `benchmark.csv` generated on representative hardware.

| Size | Threads | SIMD | NUMA | Memory Policy | FFT Schedule | FFT Tile | Mean ms | p99 ms | Throughput GB/s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 256 | 1 | auto | off | `bind-worker-buffers` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 512 | 1 | auto | off | `bind-worker-buffers` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 1024 | 1 | auto | off | `bind-worker-buffers` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 1024 | 8 | auto | auto | `bind-inputs-if-possible` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## Regression Test Results
- `ctest --output-on-failure`
- Golden regression: deterministic single-thread AVX2 checksum in `runtime_test.cpp`
- Threaded regression: multi-thread batch executor output must match single-thread output bit-exactly

## Contribution Breakdown
Update this table after each stage is measured independently on representative hardware.

| Stage | Change | Isolated Gain | Cumulative Impact |
| :--- | :--- | :--- | :--- |
| Stage 5 | FFT auto schedule + tiled filter traversal | `TBD` | `TBD` |
| Stage 2/3 | Internal thread pool + batch executor | `TBD` | `TBD` |
| Stage 6 | NUMA-aware queueing + node-local worker allocation + optional mapped-input binding | `TBD` | `TBD` |
| Stage 4 | AVX-512 f-k filter path with runtime fallback validation | `TBD` | `TBD` |

## NUMA And SIMD Reporting
`basalt_bench` now emits enough metadata to validate locality decisions alongside latency:

- `memory_policy`
- `preferred_node_counts`
- `worker_home_node_counts`
- `local_queue_executions_per_node`
- `remote_steals_from_node`

Use these fields to distinguish:

- topology-aware scheduling only
- node-local worker-buffer placement
- best-effort mapped-input binding
- remote-steal behavior under imbalance

When benchmarking AVX-512, compare checksum fields across `avx2` and `avx512` rows. On hosts without the required AVX-512 feature set, explicit `avx512` requests are expected to fall back to the AVX2 filter path.

Toolchain matrix:
- GCC/Clang: optimized AVX-512 f-k filter path is available when CPU features are present.
- MSVC or non-supporting builds: explicit AVX-512 requests degrade safely to the AVX2 path.

## Known Limitations
- AVX-512 is currently applied only to the f-k filter path.
- FFT remains AVX2-only until blocked FFT profiling shows a consistent compute-side win for AVX-512.
- NUMA-aware worker-buffer allocation is best-effort and Linux-specific.
- Mapped-input binding is opt-in via `--memory-policy bind-inputs-if-possible` and may no-op on unsupported kernels or non-Linux hosts.
- Output buffers owned by the caller are not migrated or rebound.
- No forced page migration is performed; locality comes from worker placement, node-local scratch allocation, and optional binding hints.
- PMU-based validation still requires self-hosted Linux hardware.

## Future Work
- Add self-hosted Linux benchmark baselines to populate this report with representative numbers.
- Revisit AVX-512 FFT only after post-threading and post-NUMA profiling shows the kernel is no longer memory-wall dominated.
