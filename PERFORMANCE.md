# Basalt Performance Report

Measured subject: the end-to-end f-k pipeline, `basalt::fk_filter_gather` (C++) and
`basalt.fk_filter` (Python), on float32 gathers with default parameters (pad 1.5, taper 0.1,
4-trace edge taper). Each implementation is timed doing the same work: zero-pad, real FFT,
weight multiply, inverse real FFT, crop.

## Summary

On this VM, differences under about 20% are within run-to-run noise (see Run-to-run spread),
so they are reported as ties.

- **Clearly faster than numpy at every size.** Single-threaded, Basalt is 1.6x to 3.2x faster
  than `numpy.fft.rfft2`/`irfft2` and 3.3x to 6.6x faster than `fft2`/`ifft2`.
- **Level with or faster than `scipy.fft` single-threaded.** The C++ API is ahead at 1001x1500
  (1.4x) and 1024x1024 (1.3x) and level at 240x1001 and 2048x2048 (1.15x and 1.17x, inside the
  noise). The Python binding is level at every size (1.0x to 1.2x).
- **With 4 threads, ahead of `scipy.fft` (`workers=-1`) at large sizes.** The C++ API is ahead at
  1024x1024 (1.5x) and 2048x2048 (1.7x), borderline at 1001x1500 (1.3x) and level at 240x1001.
  The Python binding is ahead at 2048x2048 (1.4x) and level elsewhere.
- **2.5x to 3.4x faster than Basalt's own legacy power-of-two path**, which was the project's
  only pipeline before 0.2.0.
- The comparison favours scipy and numpy: their weight mask is precomputed outside the timed loop,
  while Basalt computes or looks up its weights inside every call.

## Where the time goes

Basalt uses the vendored pocketfft for its transforms. In 0.2.0 the work around the transform
(padding, per-bin weights, multiply, crop) was reduced as follows. Weights are cached per
(padded size, parameters). Buffers are reused per thread and only the padding is zeroed. The
copy and multiply passes are split across threads. Single-threaded, the end-to-end call is now
within 3 to 11% of the bare pocketfft transform pair. With 4 threads it is within 7 to 50%. The
largest remaining overhead is at 1024x1024, where the transform is short enough for fixed costs
to show.

Before that change, the non-transform work took 33 to 45% of a single-thread call and 44 to 77%
of a 4-thread call. In the earlier measurement Basalt did not beat scipy reliably and lost
clearly with 4 threads.

## Results

Medians of three invocations; each invocation reports the median of 15 (Python harness) or 21
(C++ harness) timed calls after warmup. Milliseconds per gather; lower is better.

### Single thread

| Gather (padded) | Basalt C++ | Basalt Python | scipy r2c | numpy r2c | numpy c2c | legacy c2c |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 240x1001 (360x1536) | **4.1** | 4.6 | 4.7 | 7.5 | 15.4 | n/a |
| 1001x1500 (1512x2250) | **32.1** | 36.8 | 45.0 | 103.0 | 210.3 | n/a |
| 1024x1024 (1536x1536) | **22.2** | 25.0 | 28.3 | 56.3 | 146.8 | 74.7 |
| 2048x2048 (3072x3072) | **147.9** | 152.4 | 173.4 | 318.4 | 849.5 | 372.4 |

### Four threads

| Gather | Basalt C++ (`threads=4`) | Basalt Python (`threads=0`) | scipy r2c (`workers=-1`) |
| :--- | ---: | ---: | ---: |
| 240x1001 | 3.2 | 3.5 | 3.2 |
| 1001x1500 | **15.2** | 18.1 | 19.5 |
| 1024x1024 | **10.0** | 14.2 | 15.3 |
| 2048x2048 | **43.8** | 55.9 | 76.5 |

numpy has no thread option for these calls. The legacy path is single-threaded and only accepts
powers of two without padding, so it is not like-for-like; it is listed for orientation.

### Transform floor (FFT pair only, no padding or weights)

| Gather | pocketfft 1 thread | pocketfft 4 threads | scipy (ducc0) 1 thread | scipy 4 workers |
| :--- | ---: | ---: | ---: | ---: |
| 240x1001 | 4.0 | 2.6 | 5.8 | 2.1 |
| 1001x1500 | 31.2 | 13.6 | 39.8 | 11.4 |
| 1024x1024 | 21.7 | 6.6 | 23.5 | 8.2 |
| 2048x2048 | 133.5 | 40.9 | 144.1 | 52.7 |

The floors come from different harnesses (C++ with preallocated buffers vs Python allocating per
call), so compare them loosely.

### Run-to-run spread

This is a shared 4-vCPU VM. Spread is measured as (max - min) / median over the three invocations.
Two thirds of the 50 cells varied by under 20%. The worst cells (mostly 4-thread, and the smallest
gather) varied by up to about 50%, usually because of one slow invocation, which the median of three
absorbs. Raw per-invocation results are in `benchmarks/results/v0.2.0/`.

## Correctness of the comparison

Before timing, `compare_numpy.py` checks that the scipy reference and Basalt produce the same
result on a 1001x1500 gather: relative RMS difference 2.4e-7, and the precomputed numpy mask
matches `basalt.fk_mute_weight` to 5.6e-9. The 4-thread and 1-thread Basalt outputs agree to
1.6e-7.

## Hardware and software

| Field | Value |
| :--- | :--- |
| CPU | Intel Xeon @ 2.10 GHz, KVM guest, 4 vCPU, AVX2 + FMA + AVX-512 |
| NUMA | 1 node |
| OS | Linux 6.18.44 |
| Compiler | GCC 13.3.0, Release (`-O3 -mavx2 -mfma`) |
| Python | 3.13, numpy 2.5.3, scipy 1.18.1 (ducc0 backend), pybind11 3.1 |

numpy 2.x keeps float32 input in complex64, so precision does not separate Basalt from numpy or
scipy. scipy 1.18 uses ducc0, not pocketfft, so the libraries differ.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASALT_BUILD_PYTHON=ON \
      -Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')
cmake --build build -j
build/fk_gather_bench --iterations 21 --warmup 3 --csv-out fk_gather_bench.csv
python3 benchmarks/compare_numpy.py --runs 15 --warmup 3 --basalt-python build/python \
        --csv-out compare_numpy.csv
```

Pin the CPU governor and keep the machine idle when you can; this VM allowed neither.

## Known limitations

- One 4-vCPU cloud VM; numbers are indicative, not certified. Thread counts above 4 were not
  measured.
- Inputs are uniform noise. FFT cost does not depend on values.
- Transforms run unnormalised in float32. Inputs with amplitudes above about 1e31 can overflow;
  the Python binding raises `ValueError` in that case, while the C++ API returns non-finite
  values. Seismic amplitudes are far below this.
- Per-thread workspaces are kept after a call (about 113 MB per calling thread at 2048x2048) so
  repeated calls skip allocation.
- With 4 threads the Python binding is about 20 to 40% slower than the C++ API from 1001x1500 up.
  The cause (input conversion, output allocation) has not been profiled.

## Legacy path (`basalt_bench`, batch executor)

The power-of-two API (`basalt/kernel/fft2d.hpp` + `basalt/filter/fk_filter.hpp`) and its batch
executor, NUMA-aware scheduling and AVX-512 filter path are still built and tested.
`basalt_bench` sweeps size, threads, SIMD mode, NUMA mode and memory policy and emits CSV/JSON:

```bash
build/basalt_bench --sizes 256,512,1024 --threads 1,2,4 --simd auto,avx2,avx512 \
    --numa off,auto --csv-out benchmark.csv --json-out benchmark.json
```

AVX-512 applies only to that legacy filter. On hosts without AVX-512, explicit `avx512` requests
fall back to AVX2. NUMA buffer placement is best-effort and Linux-only. PMU-backed validation of
that path still needs self-hosted hardware.
