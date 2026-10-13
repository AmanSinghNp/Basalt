# Basalt Performance Report

Measured subject: the end-to-end f-k pipeline, `basalt::fk_filter_gather` (C++) and
`basalt.fk_filter` (Python), on float32 gathers with default parameters: Reject of
[-1500, 1500] m/s (a typical ground-roll mute), `dt` 4 ms, `dx` 25 m, pad 1.5, taper 0.1 and a
4-trace edge taper. Every implementation does the same work: zero-pad, real FFT, weight
multiply, inverse real FFT, crop.

## Summary

On this VM, differences under about 20% are within run-to-run noise (see Run-to-run spread).

- **2x to 4x faster than `scipy.fft`** doing the same filter: 2.1x to 3.3x single-threaded and
  1.7x to 3.8x with 4 threads (C++ API).
- **3x to 7.6x faster than `numpy.fft.rfft2`/`irfft2`**, and 5.9x to 15.6x faster than
  `fft2`/`ifft2`, single-threaded.
- **1.7x to 2.9x faster single-threaded than the first 0.2.0 pipeline** (commit 5009e1e).
- **The speed depends on the mute.** Frequency columns that the mute leaves untouched (or zeroes
  completely) skip their spatial FFTs. With the default ground-roll mute about three quarters of
  the columns are skipped. With a band that touches every column nothing is skipped, and the gain
  over 5009e1e falls to about 1.35x. Even then Basalt is about 2x faster than scipy at 1001x1500.
- **Batch API:** `fk_filter_gathers` runs many gathers in parallel. For 64 gathers of 240x1001 at
  4 threads it gives 3.1x the throughput of a loop of single-gather calls.
- **CLI end to end** (512 MB IBM SEG-Y file, read + filter + write): 2.7 s at `--threads 1` and
  1.6 s at `--threads 4`, against 4.9 s and 4.3 s for 5009e1e.
- numpy and scipy get their weight mask precomputed outside the timed loop. They do not skip
  any work based on the mask, which is where most of Basalt's lead comes from.

## How Basalt does less work

The padded FFT grid is about 2.25 times the gather (pad 1.5 on each axis). The pipeline:

1. **Skips zero rows.** It runs the time-axis real FFT only on the real traces, because the
   padded traces are all zero. It runs the inverse time-axis FFT only on the traces that are
   cropped into the output.
2. **Skips untouched columns.** The weight table is cached per (padded size, parameters) and each
   frequency column is classified once:
   - weights all exactly 1: the forward and inverse spatial FFTs would cancel, so both are skipped;
   - weights all exactly 0: the column is zeroed;
   - otherwise: the column is transformed. Several separate runs of such columns are packed into
     one transform call.
3. **Reuses buffers and splits the passes across threads.** Workspaces are reused per thread, and
   the copy and multiply passes are split across `threads`.

The result equals the plain 2D transform to float rounding (relative RMS 1.9e-7 on 60 random
configurations against the previous build). Because the zero-column shortcut would hide NaNs,
the pipeline rejects NaN and infinite input up front.

Because of this, Basalt's end-to-end call is now *faster than a bare pocketfft 2D transform
pair* of the same padded size (13.2 ms against 28.1 ms at 1001x1500, single-threaded).

## Results

Medians of three invocations; each invocation reports the median of 15 (Python harness) or 21
(C++ harness) timed calls after warmup. Milliseconds per gather; lower is better.

### Single thread

| Gather (padded) | Basalt C++ | Basalt Python | 5009e1e C++ | scipy r2c | numpy r2c | numpy c2c |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 240x1001 (360x1536) | **2.5** | 3.7 | 4.2 | 5.1 | 7.5 | 14.7 |
| 1001x1500 (1512x2250) | **13.2** | 15.0 | 30.2 | 43.3 | 100.1 | 206.1 |
| 1024x1024 (1536x1536) | **9.8** | 11.2 | 19.5 | 25.1 | 46.3 | 133.1 |
| 2048x2048 (3072x3072) | 52.5 | **48.0** | 150.8 | 162.4 | 311.3 | 700.2 |

The legacy power-of-two path (complex transform, no padding; not like-for-like) takes 66.0 ms at
1024x1024 and 321.3 ms at 2048x2048.

### Four threads

| Gather | Basalt C++ (`threads=4`) | Basalt Python (`threads=0`) | 5009e1e C++ | scipy r2c (`workers=-1`) |
| :--- | ---: | ---: | ---: | ---: |
| 240x1001 | **1.9** | 2.3 | 3.1 | 3.2 |
| 1001x1500 | 9.0 | **8.4** | 9.1 | 19.2 |
| 1024x1024 | **4.9** | 5.8 | 8.9 | 12.3 |
| 2048x2048 | 19.5 | **17.9** | 43.0 | 74.5 |

The 5009e1e 4-thread cells were noisy in this session (spread up to 64%). A separate
alternating A/B run on an idle machine gave 1.4x to 2.0x for the new build at 4 threads.

### Mute that touches every column (worst case for the column skip)

Python harness, 1001x1500, two alternating runs each:

| Band | Basalt (this build) | 5009e1e |
| :--- | ---: | ---: |
| Reject [-1500, 1500], 1 thread | 13.0 to 14.3 ms | 31.4 to 32.9 ms |
| Reject [-1e5, 1e5], 1 thread | 21.0 to 22.7 ms | 28.2 to 30.1 ms |
| Reject [-1500, 1500], 4 threads | 5.5 to 6.0 ms | 15.7 to 19.5 ms |
| Reject [-1e5, 1e5], 4 threads | 10.8 to 11.1 ms | 12.8 to 15.1 ms |

### Batch throughput (`fk_gather_bench --batch`)

| Gathers | Threads | `fk_filter_gathers` | loop of `fk_filter_gather` |
| :--- | ---: | ---: | ---: |
| 64 x 240x1001 | 1 | 157.5 ms | 158.5 ms |
| 64 x 240x1001 | 4 | **40.9 ms** | 127.4 ms |
| 16 x 1001x1500 | 1 | 229.6 ms | 246.3 ms |
| 16 x 1001x1500 | 4 | **74.2 ms** | 130.3 ms |

The batch output is bit-identical to `fk_filter_gather` with `threads = 1` on each gather,
whatever the thread count. Pool workers sleep when idle (0.3% of a core measured) and exit after
2 s without work, releasing their workspaces.

### CLI end to end

`basalt filter big.segy out.segy --gather-size 240` on a 512 MB IBM32 file (128,160 traces of 1001
samples), three alternating runs:

| Build | `--threads 1` | `--threads 4` |
| :--- | ---: | ---: |
| 5009e1e | 4.4 to 5.4 s | 4.1 to 4.5 s |
| this build | 2.0 to 3.7 s | 1.5 to 1.7 s |

The `--threads 1` and `--threads 4` outputs are byte-identical, and match 5009e1e to 2.4e-7
relative RMS.

### Run-to-run spread

This is a shared 4-vCPU VM. Spread is measured as (max - min) / median over three invocations.
Most cells varied by under 25%; some 4-thread cells and the smallest gather varied by up to about
50 to 64%, usually because of one slow invocation that the median of three absorbs. Raw results are
in `benchmarks/results/v0.2.0/`.

## Correctness of the comparison

Before timing, `compare_numpy.py` checks that the scipy reference and Basalt produce the same
result on a 1001x1500 gather (relative RMS about 2e-7), and that the precomputed numpy mask
matches `basalt.fk_mute_weight`.

## Hardware and software

| Field | Value |
| :--- | :--- |
| CPU | Intel Xeon @ 2.10 GHz, KVM guest, 4 vCPU, AVX2 + FMA + AVX-512 |
| NUMA | 1 node |
| OS | Linux 6.18.44 |
| Compiler | GCC 13.3.0, Release (`-O3 -mavx2 -mfma`) |
| Python | 3.13, numpy 2.5.3, scipy 1.18.1 (ducc0 backend), pybind11 3.1 |

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASALT_BUILD_PYTHON=ON \
      -Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')
cmake --build build -j
build/fk_gather_bench --iterations 21 --warmup 3 --csv-out fk_gather_bench.csv
build/fk_gather_bench --batch 64 --sizes 240x1001 --threads 1,4
python3 benchmarks/compare_numpy.py --runs 15 --warmup 3 --basalt-python build/python \
        --csv-out compare_numpy.csv
build/segy_io_bench --size-mb 512 --dir /tmp
```

## Options measured and rejected

These were prototyped by a swarm of agents, then re-measured independently before any decision:

| Idea | Result |
| :--- | :--- |
| AVX-512 build of the FFT with runtime dispatch | About 15% faster at 240x1001, 28% *slower* at 2048x2048 |
| FFTW single precision | 1.2x to 1.4x on some transform sizes, slower at 1512x2250; GPL licence |
| PFFFT | Slower than pocketfft at every size |
| ducc0 (scipy's backend) | Within noise of pocketfft; GPL licence |
| Fused cache-sized column slabs | No reliable gain over the column skip |
| Cheapest padded size near the target | Changes results; no reliable gain |
| pocketfft plan cache | Not reliably faster; left off |

`pad_factor` itself is the largest remaining lever: 1.25 instead of 1.5 runs about 1.5x to 1.8x
faster, with edge wrap-around about 55 dB below the input. That changes the output, so the
default stays 1.5; set it per call if your data tolerates it.

## Known limitations

- One 4-vCPU cloud VM; numbers are indicative, not certified. More than 4 threads was not measured.
- Inputs are uniform noise. FFT cost does not depend on the values, but the column skip depends on
  the mute parameters (see the worst-case table).
- Transforms run unnormalised in float32. Finite inputs with amplitudes above about 1e31 can
  overflow; the Python binding raises `ValueError`, while the C++ API returns non-finite values.
- Each calling thread keeps its workspaces after a call (about 101 MB at 2048x2048, plus up to 38
  MB when mixed columns are packed) so repeated calls skip allocation. Batch pool workers free
  theirs after 2 s idle.

## Legacy path (`basalt_bench`, batch executor)

The power-of-two API (`basalt/kernel/fft2d.hpp` + `basalt/filter/fk_filter.hpp`) and its batch
executor, NUMA-aware scheduling and AVX-512 filter path are still built and tested:

```bash
build/basalt_bench --sizes 256,512,1024 --threads 1,2,4 --simd auto,avx2,avx512 \
    --numa off,auto --csv-out benchmark.csv --json-out benchmark.json
```

AVX-512 applies only to that legacy filter, and NUMA buffer placement is best-effort and
Linux-only.
