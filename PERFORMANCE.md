# Basalt Performance Report

## Executive Summary

Measured on a 4-vCPU KVM cloud VM (Intel Xeon @ 2.10 GHz, GCC 13.3, Release, Linux 6.18). The numbers
are indicative, not certified. Release certification is still gated on representative self-hosted
Linux hardware. GitHub-hosted runners remain unsuitable for PMU-backed performance signoff.

The measured subject is the end-to-end f-k pipeline (`basalt::fk_filter_gather`, and the Python
`basalt.fk_filter` that wraps it) on float32 gathers with the default parameters.

- **Basalt is faster than numpy at every size and thread count tested.** Single-threaded, Basalt
  takes 0.19 to 0.38 of the time of `numpy.fft.fft2`/`ifft2` and 0.44 to 0.68 of the time of
  `numpy.fft.rfft2`/`irfft2`. numpy has no thread option here.
- **Basalt is not faster than `scipy.fft`.** Both are given the same padded grid and the same
  precomputed mask (for scipy). Single-threaded, Basalt takes 17 percent longer at 240x1001,
  6 percent longer at 1024x1024 and 16 percent longer at 2048x2048. The 2048x2048 gap is outside
  run-to-run spread. At 1001x1500 Basalt is 7 percent shorter, which is within spread. With 4
  threads, Basalt takes 46, 51 and 65 percent longer at 240x1001, 1024x1024 and 2048x2048, and
  the two tie at 1001x1500. The C++ harness puts Basalt behind scipy at every size and thread count.
- **The FFT library is not the reason.** Basalt's transform is the vendored pocketfft. Its
  real-to-complex and complex-to-real pair costs about the same as scipy's (see Why). The gap is in
  the work around the transform. In the C++ harness that work is 33 to 45 percent of Basalt's
  single-thread call and 44 to 77 percent of its 4-thread call.
- **Basalt scales worse with threads than scipy.** From 1 to 4 threads Basalt speeds up 1.7 to 1.8x
  at 1001x1500 and above (Python harness). scipy speeds up 2.0 to 2.5x. The transform parallelises. The surrounding
  work in `fk_filter_gather` is serial.
- **Basalt is faster than the legacy power-of-two path** (1.9x at 1024x1024 and 1.8x at 2048x2048,
  single-threaded, C++). That comparison is not like-for-like. See Legacy path comparison.
- **Two premises in the brief do not hold for these library versions.** numpy 2.5.3 keeps float32
  input in single precision (complex64), not complex128. scipy 1.18.1 computes `scipy.fft` with its
  ducc0 backend (`scipy.fft._duccfft`), not pocketfft. Float32 therefore does not separate Basalt
  from numpy or scipy, and pocketfft is not shared with scipy.

## Representative Hardware
Record benchmark hardware here for every published run.

| Field | Value |
| :--- | :--- |
| CPU model | Intel(R) Xeon(R) Processor @ 2.10 GHz (KVM guest; family 6, model 207) |
| Microarchitecture | Not reported by the VM. ISA flags include AVX2, FMA and AVX-512 (F, BW, VL, VNNI, BF16) |
| Core / thread count | 4 vCPU, 1 thread per core, 4 cores, 1 socket (as reported to the guest) |
| Caches (guest view) | L1d 48 KiB and L2 2 MiB per vCPU; L3 260 MiB reported (shared) |
| NUMA topology | 1 node (CPUs 0-3) |
| RAM / channels | 15 GiB; memory channel configuration not visible to the guest |
| OS / kernel | Linux 6.18.44-fc-v114 (`uname -r`) |
| Compiler | GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), `-O3 -DNDEBUG -mavx2 -mfma`; CMake 3.28.3 |
| Python stack | Python 3.13.16, numpy 2.5.3, scipy 1.18.1 (ducc0 backend), pybind11 3.1.0 |

The raw environment capture is in `benchmarks/results/environment.txt`. CPU frequency and governor
could not be pinned on this VM. Each invocation was started only when the 1-minute load average was
at or below 1.5. The highest load at the start of an invocation was 1.17.

## Methodology
- Use `build/Release` or equivalent optimized configuration. The measurements here used `build-ws`
  with `CMAKE_BUILD_TYPE=Release` (see Reproduce).
- Pin CPU frequency/governor before measurements on Linux. Not possible on this VM, which has no
  frequency control. Mitigations: a 1-minute load gate before each invocation, and three
  invocations per benchmark.
- Use `basalt_bench` for matrix sweeps and `fft2d_schedule_benchmark` for tile/schedule tuning.
  These are the legacy batch tools and were not run for this report.
- Keep PMU validation on self-hosted Linux with usable `perf` counters.
- Treat GitHub Actions as correctness CI, not as representative performance hardware.

### What is timed
Every timed unit is one full call on a trace-major float32 gather of `ntraces x nsamples`, filled
with uniform noise in [-1, 1] from a fixed seed. The C++ harness uses `std::mt19937` and the Python
harness uses numpy's `default_rng`, so the two harnesses see different values from the same
distribution. The FFT cost does not depend on the values. Parameters are the library defaults: `dt`
4 ms, `dx` 25 m, band [-1500, 1500] m/s, taper 0.1, `Reject`, pad factor 1.5, 4 edge-taper traces,
DC preserved.

The timed unit includes the input taper and copy, zero-padding, the forward 2D real-to-complex
transform, the mask multiply, the inverse transform and the output copy.

- **Basalt C++ (`fk_gather_bench`, `basalt-c++`)**: `basalt::fk_filter_gather`, one call per sample,
  `steady_clock`. 21 timed calls after 3 warmup calls per configuration.
- **Basalt Python (`compare_numpy.py`, `basalt-py`)**: `basalt.fk_filter`, the same calls through
  pybind11. 9 timed calls after 2 warmup calls, `perf_counter`.
- **legacy-c2c (`fk_gather_bench`)**: `fft2d_forward`, `apply_fk_filter` and `fft2d_inverse` with
  arenas, as in the README Usage section. Power-of-two sizes only, single-threaded. Each timed call
  copies the gather into the SoA arrays first. The transpose into the legacy axis order is done once,
  outside the timed region.
- **pocketfft-transform-only (`fk_gather_bench`)**: only the pipeline's own r2c and c2r calls, on the
  same padded grid and with the same pocketfft, with no taper, weight or copy work. Its buffers are
  allocated and filled once, outside the timed region. It is the floor for the transform inside
  `basalt-c++`, so the difference between the two rows is the cost of the work around it.
- **numpy (`compare_numpy.py`)**: `numpy.fft.fft2`/`ifft2` (c2c, full grid) and
  `numpy.fft.rfft2`/`irfft2` (r2c). Each timed call includes the taper, zero-padding into the
  padded grid and the output copy. The weight mask is precomputed once per size, outside the timed
  loop. This favours numpy and scipy. Basalt recomputes its weights on every call, and that cost is
  inside its timed interval.
- **scipy (`compare_numpy.py`)**: `scipy.fft.rfft2`/`irfft2` with `workers=1` and `workers=-1` (4
  threads). The timed work and the precomputed mask are the same as for numpy.
- **scipy transform-only**: the same `rfft2`/`irfft2` pair on the same padded grid, with no taper,
  mask or copy-out. Its output is not the filtered gather, so it has no relRMS value.

### Correctness checks run before timing
- The vectorised mask in `compare_numpy.py` matches `basalt.fk_mute_weight`: max absolute difference
  5.6e-9 (float32 mask) over 404 sampled bins, including the DC row and column.
- scipy r2c (`workers=1`) against `basalt.fk_filter` (`threads=1`) on 1001x1500 (padded 1512x2250):
  relRMS 2.4e-7, against a limit of 1e-4.
- Basalt with `threads=0` against `threads=1`: relRMS 1.5e-8 to 1.6e-7 across the four sizes. The
  difference is float rounding. The pipeline's two thread counts are not bit-identical, unlike the
  legacy batch executor.
- The C++ harness does not check the output. Its `checksum` column is a sanity figure only.

### Statistics and aggregation
Each invocation reports the median and nearest-rank 90th percentile of its per-call times (CSV
columns `median_ms`, `p90_ms`, `min_ms`, `mean_ms`). Three invocations of each script were run in
sequence. The tables give, per cell, the median of the three invocation medians, with the minimum
and maximum of those invocation medians in parentheses. The range is the run-to-run spread on this
VM. It is wide in some C++ cells, for example 1024x1024 at 4 threads (22.4 to 33.4 ms). A
difference that falls inside a cell's min-max range is not meaningful.

### Reproduce
Run from the repository root.

```bash
# 1. Build (Release, with the Python extension)
cmake -S . -B build-ws -DCMAKE_BUILD_TYPE=Release -DBASALT_BUILD_PYTHON=ON \
      -Dpybind11_DIR=/usr/local/lib/python3.13/dist-packages/pybind11/share/cmake/pybind11
cmake --build build-ws -j2

# 2. Environment capture (benchmarks/results/environment.txt holds the output used here)
{ lscpu; uname -r; c++ --version | head -1; cmake --version | head -1
  python3 -c "import sys, numpy, scipy, pybind11; print('python', sys.version.split()[0]); \
print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('pybind11', pybind11.__version__)"
} > benchmarks/results/environment.txt

# 3. C++ gather benchmark, three invocations (defaults: sizes 240x1001,1001x1500,1024x1024,2048x2048;
#    threads 1 and hardware concurrency; 21 timed calls and 3 warmup calls; pad 1.5)
for n in 1 2 3; do
  ./build-ws/fk_gather_bench --csv-out benchmarks/results/fk_gather_bench_run$n.csv \
      > benchmarks/results/fk_gather_bench_run$n.txt 2>&1
done

# 4. numpy, scipy and Basalt-Python comparison, three invocations (9 timed runs, 2 warmup)
for n in 1 2 3; do
  python3 benchmarks/compare_numpy.py --basalt-python build-ws/python --runs 9 --warmup 2 \
      --csv-out benchmarks/results/compare_numpy_run$n.csv > benchmarks/results/compare_numpy_run$n.txt 2>&1
done

# 5. Aggregate: median and range of median_ms across the three invocations, per configuration
python3 - <<'EOF'
import csv, glob, statistics
for prefix in ("fk_gather_bench", "compare_numpy"):
    acc = {}
    for path in sorted(glob.glob(f"benchmarks/results/{prefix}_run*.csv")):
        for r in csv.DictReader(open(path)):
            key = (r["ntraces"], r["nsamples"], r["impl"], r.get("setting", ""), r["threads"])
            acc.setdefault(key, []).append(float(r["median_ms"]))
    for key, v in sorted(acc.items()):
        print(prefix, *key, f"median {statistics.median(v):.2f}  min {min(v):.2f}  max {max(v):.2f}")
EOF

# 6. Tests
cd build-ws && ctest --output-on-failure -j2
```

Raw outputs are in `benchmarks/results/`: `fk_gather_bench_run{1,2,3}.{txt,csv}`,
`compare_numpy_run{1,2,3}.{txt,csv}` and `environment.txt`. Each CSV has a header row. The p90 values
are in the CSVs.

## Results

All times are milliseconds per gather. Each cell is the median of three invocations, with the
min-max range in parentheses. Lower is better. Padded grids are the pipeline's padded sizes.

### Single thread (threads = 1)

| Gather (NTR x NS) | Padded grid | Basalt C++ | Basalt Python | legacy-c2c | numpy c2c | numpy r2c | scipy r2c (workers=1) | scipy transform-only (ducc0, workers=1) | pocketfft transform-only (C++) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 240 x 1001 | 360 x 1536 | 6.3 (5.9-10.5) | 5.3 (5.1-5.3) | n/a | 14.1 (13.9-15.8) | 7.8 (6.7-11.1) | 4.5 (3.9-6.8) | 3.8 (3.8-4.2) | 4.2 (3.3-5.6) |
| 1001 x 1500 | 1512 x 2250 | 48.5 (44.1-61.7) | 36.9 (35.9-38.0) | n/a | 194.7 (178.6-195.5) | 83.1 (72.0-88.0) | 39.7 (37.2-45.2) | 33.6 (31.8-36.2) | 28.4 (25.9-34.2) |
| 1024 x 1024 | 1536 x 1536 | 33.2 (32.0-45.6) | 26.1 (25.4-27.0) | 63.9 (60.5-74.7) | 109.0 (102.9-133.6) | 45.5 (43.0-55.5) | 24.6 (24.2-26.5) | 23.3 (22.0-24.2) | 18.4 (18.0-25.5) |
| 2048 x 2048 | 3072 x 3072 | 176.2 (168.5-180.5) | 188.5 (183.6-189.5) | 311.8 (308.4-336.1) | 675.4 (647.1-706.5) | 303.7 (302.8-308.6) | 162.2 (157.3-168.5) | 132.9 (132.3-142.8) | 117.5 (110.2-128.1) |

### Four threads (threads = 4)
numpy is single-threaded, so its numbers are the single-thread values above.

| Gather (NTR x NS) | Basalt C++ (threads=4) | Basalt Python (threads=0, 4 CPUs) | scipy r2c (workers=-1) | scipy transform-only (ducc0, workers=-1) | pocketfft transform-only (C++, threads=4) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 240 x 1001 | 4.5 (3.9-5.3) | 4.2 (3.8-5.1) | 2.9 (2.8-3.0) | 1.8 (1.3-2.1) | 2.5 (2.4-3.2) |
| 1001 x 1500 | 29.3 (27.8-42.4) | 20.6 (17.1-23.6) | 20.3 (18.7-21.5) | 8.6 (8.0-9.5) | 8.5 (7.2-15.1) |
| 1024 x 1024 | 33.0 (22.4-33.4) | 14.8 (14.0-15.1) | 9.8 (8.7-10.7) | 7.8 (6.0-8.4) | 7.4 (5.1-8.0) |
| 2048 x 2048 | 98.8 (87.2-123.9) | 108.1 (104.5-109.3) | 65.5 (63.1-79.0) | 49.3 (48.5-54.7) | 32.6 (32.1-52.6) |

### Basalt Python time relative to each comparison
Values below 1.00 mean Basalt is faster. Ratios of the Python medians.

| Threads | Gather | vs numpy c2c | vs numpy r2c | vs scipy r2c |
| :--- | :--- | ---: | ---: | ---: |
| 1 | 240 x 1001 | 0.38 | 0.68 | 1.17 |
| 1 | 1001 x 1500 | 0.19 | 0.44 | 0.93 |
| 1 | 1024 x 1024 | 0.24 | 0.57 | 1.06 |
| 1 | 2048 x 2048 | 0.28 | 0.62 | 1.16 |
| 4 | 240 x 1001 | 0.30 | 0.54 | 1.46 |
| 4 | 1001 x 1500 | 0.11 | 0.25 | 1.02 |
| 4 | 1024 x 1024 | 0.14 | 0.32 | 1.51 |
| 4 | 2048 x 2048 | 0.16 | 0.36 | 1.65 |

### Share of each call outside its transform pair
Each share is (full call minus transform-only floor) divided by the full call, using medians of the
three invocations.

| Threads | Gather | Basalt C++ (vs pocketfft floor) | scipy r2c (vs ducc0 floor) |
| :--- | :--- | ---: | ---: |
| 1 | 240 x 1001 | 34% | 16% |
| 1 | 1001 x 1500 | 41% | 15% |
| 1 | 1024 x 1024 | 45% | 5% |
| 1 | 2048 x 2048 | 33% | 18% |
| 4 | 240 x 1001 | 44% | 36% |
| 4 | 1001 x 1500 | 71% | 57% |
| 4 | 1024 x 1024 | 77% | 20% |
| 4 | 2048 x 2048 | 67% | 25% |

The two columns come from different harnesses (C++ and Python), so compare each column with itself,
not across. The Python Basalt numbers are steadier than the C++ ones on this VM. The C++ 4-thread
cells carry the widest spread.

### Legacy path comparison (C++, single thread)
The legacy path is single-threaded and runs on power-of-two sizes only. Its work differs from the
pipeline. It does a full complex transform of the unpadded grid and uses a sigmoid mute. At
1024x1024 the pipeline transforms a 1536x1536 padded real grid, which is 2.25 times the points, and
is still 1.9 times faster than legacy-c2c (33.2 ms against 63.9 ms). At 2048x2048 the factor is 1.8
(176.2 ms against 311.8 ms). The comparison does not separate the cost of the real transform from the
cost of the legacy kernel.

### Where Basalt wins and loses
- **Wins against numpy** at every size and both thread counts. Against numpy c2c it takes 0.11 to
  0.38 of the time, and against numpy r2c 0.25 to 0.68.
- **Wins against the legacy path** at 1024x1024 and 2048x2048 (C++, 1 thread): 1.9x and 1.8x.
- **Roughly ties scipy single-threaded.** Basalt takes 17 percent longer at 240x1001, 6 percent
  longer at 1024x1024 and 16 percent longer at 2048x2048 (Python harness). The 2048x2048 gap is
  outside run-to-run spread. Basalt is 7 percent shorter at 1001x1500, which is within spread. The C++
  harness puts Basalt 9 to 40 percent behind at 1 thread.
- **Loses to scipy with 4 threads**: Basalt takes 46 to 65 percent longer at 240x1001, 1024x1024 and
  2048x2048 (Python harness), and ties at 1001x1500. The C++ harness shows the same direction at every
  size: 1.4x to 3.4x as long (the 1024x1024 cell is the noisiest).
- **Loses to scipy's transform alone** at every size and thread count, by a wide margin at 4 threads.

### Why the numbers look like this
- **The FFT library is not the cause.** Basalt's transform is the vendored pocketfft
  (`third_party/pocketfft`). scipy 1.18.1 uses ducc0 for `scipy.fft` by default, so the two do not
  share a library, contrary to the brief. On this machine Basalt's pocketfft transform pair costs
  about the same as scipy's ducc0 pair, and sometimes less: 18.4 against 23.3 ms at 1024x1024 with 1
  thread, 117.5 against 132.9 ms at 2048x2048 with 1 thread, and 32.6 against 49.3 ms at 2048x2048
  with 4 threads. At 240x1001 with 4 threads it is slower (2.5 against 1.8 ms). The two floors come
  from different harnesses, so these comparisons are indicative.
- **The work around the transform is the cause.** In the C++ harness, 33 to 45 percent of Basalt's
  single-thread call and 44 to 77 percent of its 4-thread call lies outside the pocketfft transform
  pair. For scipy the equivalent is 5 to 18 percent at 1 thread and 20 to 57 percent at 4 threads.
  Basalt's surrounding work is larger, and it is also serial.
- **The surrounding work is serial in the pipeline source.** In `src/fk/fk_pipeline.cpp`, the taper
  copy-in, the weight loop over every bin, the output copy and the two zero-filled `std::vector`
  buffers all run on one thread. The non-transform share therefore grows with thread count. The
  weight loop is the most likely largest item. It evaluates `bin_weight` for every bin on every
  call, in double precision, with a division and branches, and a cosine inside the transition bands.
  This was not isolated with a profiler.
- **numpy and scipy got a precomputed mask.** Their timed calls skip the weight evaluation that Basalt
  performs. That favours them. It is part of the work-around cost measured above.
- **numpy's real path is slower than scipy's for the same work.** At 2048x2048, single-threaded,
  numpy's full `rfft2`/`irfft2` call takes 1.9 times scipy's full call (303.7 against 162.2 ms). Both
  include the same taper, mask and copy, so the gap sits in numpy's transform path or in the
  operations around it. numpy applies its transforms one axis at a time, which typically means extra
  copies. This would explain the gap, but it was not profiled. It is why Basalt beats numpy. Basalt's
  own transform is not the reason.
- **Not explanations.** float32: numpy, scipy and Basalt all compute in float32 here. Python call
  overhead: a few microseconds per call, against milliseconds of work. The C++ and Python harnesses
  differ in both directions (C++ is slower at three of four sizes and faster at 2048x2048), which
  was not investigated. Read the two as one noisy measurement, not as a measured binding overhead.

## Regression Test Results
- `ctest --output-on-failure -j2`: 227 of 227 tests pass on this host. One GoogleTest case is
  skipped, as before: `FKFilterSimdModeTest.ExplicitAvx512FallsBackWhenUnsupported`, which is gated
  on the CPU path.
- Golden regression: deterministic single-thread AVX2 checksum in `runtime_test.cpp`.
- Threaded regression: multi-thread batch executor output must match single-thread output bit-exactly.
- The `basalt_python_tests` ctest entry runs the Python package tests.

## Legacy Results (`basalt_bench`, batch executor)
The earlier `basalt_bench` results table had placeholder values. It was not measured for this report,
so it has been removed rather than filled in. The `basalt_bench` command kept below is the legacy
batch tool, not run here. The legacy-c2c numbers above are the only legacy-path timings in this
report.

```bash
./basalt_bench --sizes 256,512,1024,2048,4096 --threads 1,2,4,8,16 --simd auto,avx2,avx512 --numa off,auto --memory-policy bind-worker-buffers,bind-inputs-if-possible --shots 64 --warmup 8 --csv-out benchmark.csv --json-out benchmark.json
./fft2d_schedule_benchmark --size 1024 --iterations 10 --warmup 2
```

`basalt_bench` writes locality counters in two formats:
- CSV: `preferred_node_counts`, `worker_home_node_counts`, `local_queue_executions_per_node`, and
  `remote_steals_from_node` are `|`-delimited flat fields.
- JSON: the same fields are emitted as numeric arrays for downstream tooling.

## Contribution Breakdown
Not measured. The per-stage table (thread pool, NUMA queueing, node-local allocation, AVX-512 filter,
FFT auto schedule) concerns the legacy batch path. Isolating those stages needs runs on self-hosted
hardware, so the placeholder table has been removed. The thread-scaling and work-around measurements
in the Results section are the only attribution in this report.

## NUMA And SIMD Reporting (legacy path)
These notes describe the legacy `basalt_bench` path. The f-k pipeline in `basalt::fk_filter_gather`
does not use the NUMA or batch-executor machinery. The notes are kept because they still describe
the legacy tooling correctly.

`basalt_bench` emits enough metadata to validate locality decisions alongside latency:

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

When benchmarking AVX-512, compare checksum fields across `avx2` and `avx512` rows. On hosts without
the required AVX-512 feature set, explicit `avx512` requests are expected to fall back to the AVX2
filter path.

Toolchain matrix (legacy path):
- GCC/Clang: optimized AVX-512 f-k filter path is available when CPU features are present.
- MSVC or non-supporting builds: explicit AVX-512 requests degrade safely to the AVX2 path.

## Known Limitations
- This is a 4-vCPU cloud VM. Numbers are indicative, not certified. CPU frequency and governor are
  not controlled. The Python harness's run-to-run range is typically 3 to 10 percent of the median.
  Some C++ cells range much more widely: 240x1001 at 1 thread spans 5.9 to 10.5 ms, and 1001x1500 at
  4 threads spans 27.8 to 42.4 ms.
- Inputs are uniform noise, not seismic data. The FFT cost does not depend on the values. The
  weight evaluation depends on them only slightly, through the transition-band branches.
- Only four sizes and two thread counts are measured. Thread counts above 4 were not run.
- The legacy path is not like-for-like with the pipeline (no padding, no edge taper, complex
  transform, sigmoid mute), so the comparison is for orientation only.
- The C++ and Python harnesses disagree by up to about 50 percent on some cells. This was not
  investigated.
- The reason for numpy's gap to scipy, and the cost of Basalt's weight loop, are not profiled.
- AVX-512 is currently applied only to the legacy f-k filter path.
- FFT remains AVX2-only until blocked FFT profiling shows a consistent compute-side win for AVX-512.
- NUMA-aware worker-buffer allocation is best-effort and Linux-specific.
- Mapped-input binding is opt-in via `--memory-policy bind-inputs-if-possible` and may no-op on
  unsupported kernels or non-Linux hosts.
- Output buffers owned by the caller are not migrated or rebound.
- No forced page migration is performed; locality comes from worker placement, node-local scratch
  allocation, and optional binding hints.
- PMU-based validation still requires self-hosted Linux hardware.

## Future Work
- Cache the weight grid for each (padded size, parameters) pair, so repeated calls skip per-bin
  weight evaluation. Then re-measure the work-around share.
- Profile the weight loop and confirm it is the largest non-transform item.
- Parallelise the weight loop, the taper copy-in and the output copy inside `fk_filter_gather`, and
  check whether the non-transform share at 4 threads falls.
- Profile numpy's real transform path to confirm the axis-by-axis explanation.
- Add self-hosted Linux benchmark baselines to populate this report with representative numbers.
- Revisit AVX-512 FFT only after post-threading and post-NUMA profiling shows the kernel is no longer
  memory-wall dominated.
