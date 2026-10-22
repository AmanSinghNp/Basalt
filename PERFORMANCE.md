# Basalt Performance Report

## Executive Summary
Basalt now has the core scaffolding needed to validate optimization stages end to end:

- cache-aware FFT schedule selection is wired into the default path
- the f-k filter supports tiled traversal and runtime SIMD dispatch
- an internal batch executor, thread pool, NUMA topology detector, and node-local worker-buffer allocator exist for throughput sweeps
- the benchmark harness can sweep size, thread count, SIMD mode, NUMA mode, and memory policy and emit CSV/JSON

The measurements below were taken on a 4-vCPU shared cloud VM with a single NUMA node. They confirm the functional path and give indicative numbers. They are not representative of the self-hosted Linux hardware that Release certification requires. The main measured results are: the `row-first-column-strips` FFT schedule (selected by `auto`) is 1.47x faster than the best transpose schedule at 1024x1024 and 2.03x faster at 2048x2048 (forward 2D FFT, min over 5 runs); the AVX2 and AVX-512 f-k paths produce identical checksums; and the Basalt single-thread pipeline is faster than the NumPy reference at 256, 512 and 1024. Thread scaling was not measurable on this host.

GitHub-hosted runners remain unsuitable for PMU-backed performance signoff.

## Environment
Measured on this host (values from `/proc/cpuinfo`, `lscpu`, `free`, `uname`, `g++`):

| Item | Value |
| :--- | :--- |
| Commit | `40a304d` (`claude/dazzling-mendel-3ue804`) |
| CPU | Intel(R) Xeon(R) Processor @ 2.10GHz, KVM guest, CPUID family 6 model 207 (0xCF) |
| Cores | 4 vCPUs, 1 thread per core as reported by the guest, 4 CPUs in `NUMA node0` |
| Caches (per lscpu) | L1d 48 KiB and L2 2 MiB per core; L3 260 MiB shared |
| SIMD flags | AVX2, FMA, AVX-512F/DQ/BW/VL present |
| RAM | 16 GB |
| Kernel / OS | Linux 6.18.44-fc-v140, Ubuntu 24.04.5 LTS |
| Compiler | `/usr/bin/c++` = g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| Build flags | `CMAKE_BUILD_TYPE=Release` (default): `-O3 -DNDEBUG -std=c++17 -mtune=generic -Wall -Wextra -Wpedantic -Wshadow -Wconversion`; no `-march=native` |
| Build system | CMake with Ninja, default options |
| Frequency governor | Not pinned: the guest exposes no cpufreq sysfs interface |
| Concurrent load | Other processes shared the 4 vCPUs during the runs (1-minute load average 1.6 to 5.3) |
| PMU counters | Not collected: `perf` is not installed on this host |
| Python / NumPy | Python 3.13.16, NumPy 2.5.3 |

## Representative Hardware
Record benchmark hardware here for every published run.

| Field | Value |
| :--- | :--- |
| CPU model | Intel(R) Xeon(R) Processor @ 2.10GHz (KVM guest) |
| Microarchitecture | Not reported by the guest; CPUID family 6, model 207 (0xCF) |
| Core / thread count | 4 vCPUs, 1 thread per core (as reported by the guest) |
| NUMA topology | 1 node (node0, CPUs 0-3) |
| RAM / channels | 16 GB; channel count not exposed by the guest |
| OS / kernel | Ubuntu 24.04.5 LTS / Linux 6.18.44-fc-v140 |
| Compiler | g++ 13.3.0 |

This host is a shared cloud VM. It is not the self-hosted Linux hardware required for Release signoff.

## Methodology
- Use `build/Release` or equivalent optimized configuration.
- Pin CPU frequency/governor before measurements on Linux.
- Use `basalt_bench` for matrix sweeps and `fft2d_schedule_benchmark` for tile/schedule tuning.
- Keep PMU validation on self-hosted Linux with usable `perf` counters.
- Treat GitHub Actions as correctness CI, not as representative performance hardware.

How the numbers in this report were produced on this host:
- Each `basalt_bench` configuration was run 5 times in sequence (same seed 1337, 3 warmup shots and 20 timed shots per run). The table reports the median of the per-run `mean_ms`, the median of the per-run `p99_ms`, and the median of the per-run `throughput_gb_s`. The bracketed range in the Mean ms column is the min to max over the 5 runs.
- `mean_ms` and `p99_ms` are per-shot latencies of one `BatchExecutor::run` call: forward 2D FFT, f-k band mute, inverse 2D FFT.
- `throughput_gb_s` is the benchmark's nominal figure, `size^2 * 4 bytes * 2 * shots / elapsed time`. It is not a measured memory bandwidth.
- `fft2d_schedule_benchmark` was run 5 times per size. Candidate timings are per-run minima, and the table reports the minimum over the 5 runs. It times the forward transform only.
- `baseline.py` was run 5 times. Its mean and best per-shot times are reported as the median over runs, with the min to max range in brackets.
- The run-to-run spread on this host is large: the min-max range within one configuration is often 10 to 40 percent of the median and reaches about 70 percent for some small-size rows. Differences smaller than that spread should not be read as gains.

### Benchmark Commands
```bash
./basalt_bench --sizes 256,512,1024,2048,4096 --threads 1,2,4,8,16 --simd auto,avx2,avx512 --numa off,auto --memory-policy bind-worker-buffers,bind-inputs-if-possible --shots 64 --warmup 8 --csv-out benchmark.csv --json-out benchmark.json
./fft2d_schedule_benchmark --size 1024 --iterations 10 --warmup 2
```

`basalt_bench` writes locality counters in two formats:
- CSV: `preferred_node_counts`, `worker_home_node_counts`, `local_queue_executions_per_node`, and `remote_steals_from_node` are `|`-delimited flat fields.
- JSON: the same fields are emitted as numeric arrays for downstream tooling.

## Results Table
Populated from 5 repeated runs of `basalt_bench` on the host described in Environment (not representative hardware). Mean ms is the median with the min-max range in brackets; p99 ms and throughput are medians.

Every row uses `row-first-column-strips` as the resolved FFT schedule. The `FFT Tile` value of 128 is the resolved default; the strips schedule ignores the tile parameter. The `--numa off` rows run with a single NUMA node, so NUMA placement has no effect here. Thread counts above 1 were run on 4 vCPUs shared with other processes.

| Size | Threads | SIMD | NUMA | Memory Policy | FFT Schedule | FFT Tile | Mean ms | p99 ms | Throughput GB/s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 256 | 1 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 1.19 [1.15-1.93] | 1.56 | 0.44 |
| 256 | 1 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 1.05 [0.94-1.41] | 1.49 | 0.50 |
| 256 | 1 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 0.72 [0.70-1.12] | 0.96 | 0.72 |
| 256 | 2 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 1.63 [1.20-1.88] | 2.09 | 0.51 |
| 256 | 2 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 1.11 [0.99-1.27] | 1.38 | 0.75 |
| 256 | 2 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 0.94 [0.76-1.25] | 1.17 | 0.87 |
| 256 | 4 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 1.49 [1.23-2.10] | 2.08 | 0.91 |
| 256 | 4 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 1.16 [0.96-1.25] | 1.48 | 1.31 |
| 256 | 4 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 1.00 [0.78-1.00] | 1.25 | 1.50 |
| 512 | 1 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 5.70 [5.37-6.27] | 7.61 | 0.37 |
| 512 | 1 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 4.88 [4.43-5.93] | 6.11 | 0.43 |
| 512 | 1 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 3.58 [3.56-4.94] | 4.54 | 0.58 |
| 512 | 2 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 7.69 [5.46-9.55] | 8.27 | 0.52 |
| 512 | 2 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 5.27 [4.70-7.91] | 7.25 | 0.71 |
| 512 | 2 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 4.73 [3.69-5.43] | 4.97 | 0.86 |
| 512 | 4 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 7.17 [5.65-11.44] | 9.91 | 0.99 |
| 512 | 4 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 5.47 [4.69-9.51] | 6.13 | 1.27 |
| 512 | 4 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 4.43 [3.56-8.17] | 5.25 | 1.55 |
| 1024 | 1 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 25.00 [22.08-33.84] | 34.81 | 0.34 |
| 1024 | 1 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 20.68 [19.26-24.50] | 24.99 | 0.41 |
| 1024 | 1 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 19.76 [15.91-21.04] | 21.52 | 0.42 |
| 1024 | 2 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 25.55 [24.23-38.73] | 34.12 | 0.60 |
| 1024 | 2 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 22.71 [19.79-26.38] | 29.71 | 0.69 |
| 1024 | 2 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 21.22 [15.60-24.70] | 24.07 | 0.77 |
| 1024 | 4 | scalar | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 35.85 [25.65-49.39] | 43.86 | 0.85 |
| 1024 | 4 | avx2 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 28.92 [20.82-39.94] | 71.32 | 0.88 |
| 1024 | 4 | avx512 | off | `bind-worker-buffers` | `row-first-column-strips` | 128 | 20.56 [16.21-33.66] | 33.63 | 1.39 |
| 1024 | 4 | auto | auto | `bind-inputs-if-possible` | `row-first-column-strips` | 128 | 20.14 [17.03-31.96] | 26.77 | 1.40 |

The last row is the NUMA-auto configuration. It replaces the earlier 8-thread row because this host has 4 vCPUs. On a single-node host the NUMA-auto path has no topology to exploit, so the row is a functional check only.

Checksums: the `checksum_real` and `checksum_imag` fields are identical across all runs and thread counts for each SIMD mode. The AVX2 and AVX-512 checksums are identical at every size and thread count. The scalar checksums differ from the AVX2 and AVX-512 checksums at every size. This is expected: the scalar kernels evaluate the taper sigmoid with `std::exp` while the SIMD paths use the `fast_exp_avx` polynomial (weights agree within 1e-5), and the scalar FFT accumulates in a different order, so outputs agree to float tolerance but not bit for bit. The tests compare all three paths against double-precision references.

### FFT Schedule Comparison (forward 2D FFT)
`fft2d_schedule_benchmark`, 10 iterations, 2 warmup shots, 3 interleaved rounds, 5 runs. Each cell is the minimum over the 5 runs of the per-run minimum.

| Schedule | Tile | 1024x1024 min ms | 2048x2048 min ms |
| :--- | :--- | :--- | :--- |
| `legacy-column-first` | 64 | 7.706 | 47.266 |
| `legacy-column-first` | 96 | 7.772 | 47.680 |
| `legacy-column-first` | 128 | 7.647 | 49.372 |
| `legacy-column-first` | 192 | 7.695 | 48.697 |
| `four-step-row-first` | 64 | 7.703 | 48.132 |
| `four-step-row-first` | 96 | 7.817 | 51.041 |
| `four-step-row-first` | 128 | 7.653 | 49.350 |
| `four-step-row-first` | 192 | 7.718 | 52.189 |
| `row-first-column-strips` | n/a | 5.214 | 23.285 |
| `auto` | n/a | 5.194 | 23.185 |

`auto` resolves to `row-first-column-strips` at both sizes. The best strips-to-transpose ratio is 1.47x at 1024 and 2.03x at 2048. The largest absolute difference between any candidate and the first candidate was 7.3e-4 at 1024 (values up to about 3.0e3) and 1.5e-3 at 2048 (values up to about 6.2e3). These are float32 rounding differences.

### NumPy Reference (same sizes)
`benchmarks/baseline.py --size 256 512 1024 --iters 20 --warmup 3 --seed 1337`, 5 runs. The NumPy pipeline is `fft2 -> band mute -> ifft2` in complex64, the same work as one `basalt_bench` shot. Basalt columns are medians of the sweep table above. NumPy's FFT is not multithreaded by default; this was not measured separately.

| Size | NumPy mean ms (median, range) | NumPy best ms (median) | Basalt 1 thread avx2 mean ms | Basalt 1 thread avx512 mean ms | Basalt 4 threads avx512 mean ms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 256 | 3.05 [2.30-3.47] | 1.80 | 1.05 | 0.72 | 1.00 |
| 512 | 9.81 [9.15-10.59] | 7.44 | 4.88 | 3.58 | 4.43 |
| 1024 | 51.36 [45.39-60.77] | 38.67 | 20.68 | 19.76 | 20.56 |

## Regression Test Results
- `ctest --output-on-failure`
- Pipeline regression: batch executor output must match the direct single-call pipeline within a relative tolerance (`runtime_test.cpp`), plus double-precision reference checks for FFT and filter
- Threaded regression: multi-thread batch executor output must match single-thread output bit-exactly

## Contribution Breakdown
Update this table after each stage is measured independently on representative hardware. The values below come from the shared 4-vCPU host in Environment and are indicative only.

| Stage | Change | Isolated Gain | Cumulative Impact |
| :--- | :--- | :--- | :--- |
| Stage 5 | FFT auto schedule + tiled filter traversal | Forward 2D FFT: `row-first-column-strips` (selected by `auto`) is 1.47x faster than the best legacy or four-step schedule at 1024x1024 and 2.03x at 2048x2048. Filter-traversal gain not measured separately. | Not measured separately from the full pipeline. |
| Stage 2/3 | Internal thread pool + batch executor | Not measurable on this host. Scalar 1024x1024 mean was 25.00 ms at 1 thread and 35.85 ms at 4 threads (medians), with other processes sharing the vCPUs. | Not measured. |
| Stage 6 | NUMA-aware queueing + node-local worker allocation + optional mapped-input binding | Not measurable: the host has one NUMA node. | Not measured. |
| Stage 4 | AVX-512 f-k filter path with runtime fallback validation | Single thread, median mean ms, AVX2 to AVX-512: 1.05 to 0.72 at 256 (1.45x), 4.88 to 3.58 at 512 (1.36x), 20.68 to 19.76 at 1024 (1.05x; ranges overlap at 1024). | Not measured separately. Checksums match AVX2 at every size. |

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

On this host, every run reports `numa_nodes=1`, and the NUMA-auto run recorded `preferred_node_counts=4`, `worker_home_node_counts=4`, `local_queue_executions_per_node=20`, and `remote_steals_from_node=0`. Remote-steal behavior under imbalance cannot be assessed on one node.

When benchmarking AVX-512, compare checksum fields across `avx2` and `avx512` rows. On hosts without the required AVX-512 feature set, explicit `avx512` requests are expected to fall back to the AVX2 filter path.

Toolchain matrix:
- GCC/Clang: optimized AVX-512 f-k filter path is available when CPU features are present.
- MSVC: the AVX-512 path is selected through CPUID/XGETBV detection (not yet exercised in CI).
- CPUs without AVX-512F/DQ: explicit AVX-512 requests degrade to the AVX2 path, and CPUs without AVX2/FMA use the scalar kernels. `BASALT_SIMD_MAX=scalar|avx2|avx512` caps the path for testing.

## How To Reproduce
From the repository root, on a host with the Environment above:

```bash
cmake -S . -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel

# Matrix sweep (run 5 times; report medians)
./build-rel/basalt_bench --sizes 256,512,1024 --threads 1,2,4 --simd scalar,avx2,avx512 --numa off --memory-policy bind-worker-buffers --shots 20 --warmup 3 --seed 1337 --csv-out sweep.csv --json-out sweep.json

# NUMA-auto check at 4 threads (run 5 times)
./build-rel/basalt_bench --sizes 1024 --threads 4 --simd auto --numa auto --memory-policy bind-inputs-if-possible --shots 20 --warmup 3 --seed 1337 --csv-out numa.csv --json-out numa.json

# FFT schedule comparison (run 5 times; report minimum)
./build-rel/fft2d_schedule_benchmark --size 1024 --iterations 10 --warmup 2 --rounds 3
./build-rel/fft2d_schedule_benchmark --size 2048 --iterations 10 --warmup 2 --rounds 3

# NumPy reference (run 5 times)
python3 benchmarks/baseline.py --size 256 512 1024 --iters 20 --warmup 3 --seed 1337
```

The Stage 1 profiling workflow also collects `perf` counters (`tools/profiling/run_stage1.sh`). It was not run here because `perf` is not installed on this host. The raw CSV and JSON files are not committed; regenerate them with the commands above.

## Known Limitations
- AVX-512 is currently applied only to the f-k filter path.
- FFT remains AVX2-only until blocked FFT profiling shows a consistent compute-side win for AVX-512.
- NUMA-aware worker-buffer allocation is best-effort and Linux-specific.
- Mapped-input binding is opt-in via `--memory-policy bind-inputs-if-possible` and may no-op on unsupported kernels or non-Linux hosts.
- Output buffers owned by the caller are not migrated or rebound.
- No forced page migration is performed; locality comes from worker placement, node-local scratch allocation, and optional binding hints.
- PMU-based validation still requires self-hosted Linux hardware.
- The shared 4-vCPU host used for this report cannot show thread scaling or NUMA effects, and its run-to-run noise is large.

## Future Work
- Add self-hosted Linux benchmark baselines to populate this report with representative numbers.
- Revisit AVX-512 FFT only after post-threading and post-NUMA profiling shows the kernel is no longer memory-wall dominated.
