#!/usr/bin/env python3
"""Times Basalt's f-k pipeline against numpy.fft and scipy.fft doing the same work.

Each timed call runs the full pipeline of docs/design/FK_PIPELINE.md on one trace-major float32
gather: edge taper, zero-padding to the pipeline's padded sizes, forward 2D transform, multiply by
the velocity-band weight mask, inverse 2D transform and copy-out of the valid block.

Cases:
  numpy-c2c    numpy.fft.fft2 / ifft2 on the full complex grid, real part kept
  numpy-r2c    numpy.fft.rfft2 / irfft2 (half spectrum, like the pipeline)
  scipy-r2c    scipy.fft.rfft2 / irfft2 with workers=1 and workers=-1
  basalt-py    basalt.fk_filter with threads=1 and threads=0 (hardware concurrency)
  scipy-transform-only
               scipy.fft rfft2 / irfft2 alone on the padded grid, with no taper, mask or
               copy: the floor for the transform. Its output is not the filtered gather.

The weight mask is computed once per size, outside the timed loop, for numpy and scipy. This
favours them: basalt.fk_filter recomputes its weights inside every call, and that cost is timed.

Precision: numpy 2.x keeps float32 input in single precision (complex64), and so does scipy.fft.
All four families therefore run in float32 here. The numpy dtypes are printed at start-up.

Before timing, the scipy r2c result is checked against basalt.fk_filter on the verification size
(relRMS must be <= 1e-4), and the numpy mask is checked against basalt.fk_mute_weight.

Each case reports the median and 90th percentile of at least 7 timed runs after warmup. The
timed interval covers the whole call, so Python call overhead is included on both sides.
Contract: docs/design/FK_PIPELINE.md (sections 2-4 and 7).

Usage (after building the Python extension, see PERFORMANCE.md):
  python3 benchmarks/compare_numpy.py --basalt-python build-ws/python
"""

import argparse
import csv
import gc
import math
import os
import platform
import random
import statistics
import sys
import time

import numpy as np
import scipy
import scipy.fft as sfft

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SIZES = "240x1001,1001x1500,1024x1024,2048x2048"
DEFAULT_CSV = os.path.join(REPO_ROOT, "benchmarks", "results", "compare_numpy.csv")
TRACE_PRIMES = (2, 3, 5, 7, 11)   # pocketfft good_size_cmplx, used for the trace axis
SAMPLE_PRIMES = (2, 3, 5)         # pocketfft good_size_real, used for the sample axis

# Pipeline defaults (basalt::FKMuteParams), shared by every case.
DT = 0.004
DX = 25.0
V_MIN = -1500.0
V_MAX = 1500.0
TAPER = 0.1
EDGE_TAPER_TRACES = 4
EDGE_TAPER_SAMPLES = 0
PRESERVE_DC = True
SEED = 20260101
RELRMS_LIMIT = 1e-4


def parse_size(text):
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise argparse.ArgumentTypeError(f"size must be NTRxNS, got '{text}'")
    return int(parts[0]), int(parts[1])


def parse_sizes(text):
    return [parse_size(item) for item in text.split(",") if item.strip()]


def smooth_at_least(n, primes):
    """Smallest integer >= n whose prime factors all lie in `primes`."""
    m = max(n, 1)
    while True:
        x = m
        for p in primes:
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1


def padded_sizes(ntr, ns, pad_factor):
    """Padded sizes of section 4, step 2: good_size(ceil(n * pad_factor)) per axis."""
    ntr_pad = smooth_at_least(math.ceil(ntr * pad_factor), TRACE_PRIMES)
    ns_pad = smooth_at_least(math.ceil(ns * pad_factor), SAMPLE_PRIMES)
    return ntr_pad, ns_pad


def edge_weights(n, width):
    """Hann half-window taper on both ends of a line of n samples (section 4, step 3)."""
    w = np.ones(n, dtype=np.float64)
    m = min(width, n)
    i = np.arange(m)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (width + 1.0))
    w[:m] *= ramp
    w[n - 1 - i] *= ramp
    return w


def _rise(v, edge, half):
    """Raised-cosine step of section 3, vectorised. v may be +inf."""
    if half == 0.0:
        return (v > edge).astype(np.float64)
    vc = np.clip(v, edge - half, edge + half)
    return 0.5 - 0.5 * np.cos(np.pi * (vc - (edge - half)) / (2.0 * half))


def mute_weight(f_hz, k_cpm):
    """Section 3 weight (Reject mode) for broadcastable frequency and wavenumber arrays.

    Same definition as basalt::fk_mute_weight: v = -f / k, v = +inf for k = 0 (f != 0), and the
    DC bin is forced to 1 because preserve_dc is set.
    """
    f, k = np.broadcast_arrays(np.asarray(f_hz, dtype=np.float64),
                               np.asarray(k_cpm, dtype=np.float64))
    k_zero = k == 0.0
    dc = k_zero & (f == 0.0)
    v = np.where(k_zero, np.inf, -f / np.where(k_zero, 1.0, k))
    v = np.where(dc, 0.0, v)
    inside = (_rise(v, V_MIN, 0.5 * TAPER * abs(V_MIN))
              * (1.0 - _rise(v, V_MAX, 0.5 * TAPER * abs(V_MAX))))
    weight = 1.0 - inside
    if PRESERVE_DC:
        weight = np.where(dc, 1.0, weight)
    return weight


def make_masks(ntr_pad, ns_pad):
    """Half-spectrum mask for the r2c paths and full-grid mask for the c2c path (float32)."""
    nfreq = ns_pad // 2 + 1
    f_half = np.arange(nfreq) / (ns_pad * DT)
    k = np.fft.fftfreq(ntr_pad, d=DX)
    mask_half = mute_weight(f_half[None, :], k[:, None]).astype(np.float32)
    f_full = np.fft.fftfreq(ns_pad, d=DT)
    mask_full = mute_weight(f_full[None, :], k[:, None]).astype(np.float32)
    return mask_half, mask_full


def make_gather(ntr, ns):
    rng = np.random.default_rng(SEED)
    return rng.uniform(-1.0, 1.0, size=(ntr, ns)).astype(np.float32)


def relrms(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    return float(np.sqrt(np.mean((a - e) ** 2)) / np.sqrt(np.mean(e ** 2)))


def percentile90(values):
    """Nearest-rank 90th percentile (same rule as fk_gather_bench)."""
    ordered = sorted(values)
    rank = max(math.ceil(0.9 * len(ordered)), 1)
    return ordered[min(rank, len(ordered)) - 1]


def time_case(fn, runs, warmup):
    for _ in range(warmup):
        fn()
    gc.collect()
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
        out = result
    return times, out


def check_mask(basalt, ntr_pad, ns_pad, mask_full, count=400):
    """Compares the vectorised mask with basalt.fk_mute_weight at sampled and special bins."""
    f_axis = np.fft.fftfreq(ns_pad, d=DT)
    k_axis = np.fft.fftfreq(ntr_pad, d=DX)
    rng = random.Random(1)
    bins = [(0, 0), (0, 1), (1, 0), (ntr_pad // 2, ns_pad // 2)]
    bins += [(rng.randrange(ntr_pad), rng.randrange(ns_pad)) for _ in range(count)]
    worst = 0.0
    for i, j in bins:
        ref = basalt.fk_mute_weight(float(f_axis[j]), float(k_axis[i]),
                                    v_min=V_MIN, v_max=V_MAX, taper=TAPER, mode="reject")
        worst = max(worst, abs(ref - float(mask_full[i, j])))
    return worst


def bench_size(basalt, ntr, ns, args, ncpu):
    x = make_gather(ntr, ns)
    ntr_pad, ns_pad = padded_sizes(ntr, ns, args.pad)
    shape = (ntr_pad, ns_pad)
    wt = edge_weights(ntr, EDGE_TAPER_TRACES).astype(np.float32)
    ws = edge_weights(ns, EDGE_TAPER_SAMPLES).astype(np.float32)
    mask_half, mask_full = make_masks(ntr_pad, ns_pad)
    bkw = dict(v_min=V_MIN, v_max=V_MAX, taper=TAPER, mode="reject", pad_factor=args.pad,
               edge_taper_traces=EDGE_TAPER_TRACES, edge_taper_samples=EDGE_TAPER_SAMPLES,
               preserve_dc=PRESERVE_DC)

    def padded_input():
        xp = np.zeros(shape, dtype=np.float32)
        xp[:ntr, :ns] = x * wt[:, None] * ws[None, :]
        return xp

    def numpy_c2c():
        spec = np.fft.fft2(padded_input())
        spec *= mask_full
        y = np.fft.ifft2(spec)
        return y.real[:ntr, :ns].copy()

    def numpy_r2c():
        spec = np.fft.rfft2(padded_input())
        spec *= mask_half
        y = np.fft.irfft2(spec, s=shape)
        return y[:ntr, :ns].copy()

    def scipy_r2c(workers):
        def run():
            spec = sfft.rfft2(padded_input(), workers=workers)
            spec *= mask_half
            y = sfft.irfft2(spec, s=shape, workers=workers)
            return y[:ntr, :ns].copy()
        return run

    def basalt_run(threads):
        return lambda: basalt.fk_filter(x, DT, DX, threads=threads, **bkw)

    # Floor for the transform pair alone: no taper, pad, mask or copy-out, same padded grid.
    # Its output is not the filtered gather, so it has no relRMS column.
    xp_fixed = padded_input()

    def transform_only(workers):
        def run():
            spec = sfft.rfft2(xp_fixed, workers=workers)
            return sfft.irfft2(spec, s=shape, workers=workers)
        return run

    cases = [
        ("numpy-c2c", "numpy.fft fft2/ifft2", 1, numpy_c2c, True),
        ("numpy-r2c", "numpy.fft rfft2/irfft2", 1, numpy_r2c, True),
        ("scipy-r2c", "scipy.fft workers=1", 1, scipy_r2c(1), True),
        ("scipy-r2c", "scipy.fft workers=-1", ncpu, scipy_r2c(-1), True),
        ("basalt-py", "basalt.fk_filter threads=1", 1, basalt_run(1), True),
        ("basalt-py", "basalt.fk_filter threads=0", ncpu, basalt_run(0), True),
        ("scipy-transform-only", "rfft2+irfft2 workers=1", 1, transform_only(1), False),
        ("scipy-transform-only", "rfft2+irfft2 workers=-1", ncpu, transform_only(-1), False),
    ]

    results = []
    for impl, setting, threads, fn, compare in cases:
        times, out = time_case(fn, args.runs, args.warmup)
        results.append({
            "_compare": compare,
            "impl": impl,
            "setting": setting,
            "ntraces": ntr,
            "nsamples": ns,
            "ntraces_padded": ntr_pad,
            "nsamples_padded": ns_pad,
            "threads": threads,
            "runs": args.runs,
            "warmup": args.warmup,
            "pad_factor": args.pad,
            "median_ms": statistics.median(times),
            "p90_ms": percentile90(times),
            "min_ms": min(times),
            "mean_ms": statistics.fmean(times),
            "gathers_per_s": 1000.0 / statistics.median(times),
            "_out": out,
        })
        print(f"  {ntr}x{ns} {impl:<10} {setting:<26} median {results[-1]['median_ms']:9.3f} ms",
              flush=True)

    reference = next(r["_out"] for r in results if r["impl"] == "basalt-py"
                     and r["threads"] == 1)
    for r in results:
        compare = r.pop("_compare")
        out = r.pop("_out")
        r["relrms_vs_basalt_t1"] = relrms(out, reference) if compare else None
    return results


def verify(basalt, ntr, ns, args):
    """Numerical check before any timing: scipy r2c against basalt on one size."""
    x = make_gather(ntr, ns)
    ntr_pad, ns_pad = padded_sizes(ntr, ns, args.pad)
    shape = (ntr_pad, ns_pad)
    wt = edge_weights(ntr, EDGE_TAPER_TRACES).astype(np.float32)
    ws = edge_weights(ns, EDGE_TAPER_SAMPLES).astype(np.float32)
    mask_half, mask_full = make_masks(ntr_pad, ns_pad)

    xp = np.zeros(shape, dtype=np.float32)
    xp[:ntr, :ns] = x * wt[:, None] * ws[None, :]
    spec = sfft.rfft2(xp, workers=1)
    spec *= mask_half
    scipy_out = sfft.irfft2(spec, s=shape, workers=1)[:ntr, :ns]

    bkw = dict(v_min=V_MIN, v_max=V_MAX, taper=TAPER, mode="reject", pad_factor=args.pad,
               edge_taper_traces=EDGE_TAPER_TRACES, edge_taper_samples=EDGE_TAPER_SAMPLES,
               preserve_dc=PRESERVE_DC)
    basalt_t1 = basalt.fk_filter(x, DT, DX, threads=1, **bkw)
    basalt_t0 = basalt.fk_filter(x, DT, DX, threads=0, **bkw)

    err_scipy = relrms(scipy_out, basalt_t1)
    mask_err = check_mask(basalt, ntr_pad, ns_pad, mask_full)
    print(f"verify {ntr}x{ns} (padded {ntr_pad}x{ns_pad}):")
    print(f"  mask vs basalt.fk_mute_weight: max abs diff {mask_err:.3e} (float32 mask)")
    print(f"  scipy-r2c (workers=1) vs basalt.fk_filter (threads=1): relRMS {err_scipy:.3e}")
    print(f"  basalt threads=0 vs threads=1: relRMS {relrms(basalt_t0, basalt_t1):.3e}")
    if not err_scipy <= RELRMS_LIMIT:
        raise SystemExit(f"verification failed: relRMS {err_scipy:.3e} > {RELRMS_LIMIT:.0e}")
    if not mask_err <= 1e-6:
        raise SystemExit(f"verification failed: mask differs from fk_mute_weight by {mask_err:.3e}")


def print_environment(basalt, args):
    print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"basalt {basalt.__version__}")
    probe = np.zeros((2, 2), dtype=np.float32)
    print(f"dtypes: numpy.fft.fft2(float32) -> {np.fft.fft2(probe).dtype}, "
          f"numpy.fft.rfft2(float32) -> {np.fft.rfft2(probe).dtype}, "
          f"scipy.fft.rfft2(float32) -> {sfft.rfft2(probe).dtype}")
    print(f"cpus {os.cpu_count()}, runs {args.runs}, warmup {args.warmup}, pad {args.pad}")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["impl", "setting", "ntraces", "nsamples", "ntraces_padded", "nsamples_padded",
              "threads", "runs", "warmup", "pad_factor", "median_ms", "p90_ms", "min_ms",
              "mean_ms", "gathers_per_s", "relrms_vs_basalt_t1"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v)
                             for k, v in row.items() if k in fields})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default=DEFAULT_SIZES,
                    help="comma-separated NTRxNS gathers (default: %(default)s)")
    ap.add_argument("--verify-size", default="1001x1500", type=parse_size,
                    help="size for the numerical check before timing (default: %(default)s)")
    ap.add_argument("--runs", type=int, default=9, help="timed runs per case, >= 7 (default 9)")
    ap.add_argument("--warmup", type=int, default=2, help="untimed runs per case (default 2)")
    ap.add_argument("--pad", type=float, default=1.5, help="pad_factor (default 1.5)")
    ap.add_argument("--basalt-python", default=os.path.join(REPO_ROOT, "build", "python"),
                    help="directory containing the basalt package (default: %(default)s)")
    ap.add_argument("--csv-out", default=DEFAULT_CSV, help="CSV output path (default: %(default)s)")
    args = ap.parse_args()
    if args.runs < 7:
        ap.error("--runs must be at least 7")
    if args.warmup < 0 or args.pad < 1.0:
        ap.error("--warmup must be >= 0 and --pad >= 1")
    try:
        sizes = parse_sizes(args.sizes)
    except argparse.ArgumentTypeError as err:
        ap.error(str(err))
    if not sizes:
        ap.error("--sizes is empty")

    sys.path.insert(0, os.path.abspath(args.basalt_python))
    try:
        import basalt
    except ImportError as err:
        raise SystemExit(f"cannot import basalt from {args.basalt_python}: {err}")

    ncpu = os.cpu_count() or 1
    print_environment(basalt, args)
    verify(basalt, args.verify_size[0], args.verify_size[1], args)

    rows = []
    for ntr, ns in sizes:
        print(f"timing {ntr}x{ns}", flush=True)
        rows.extend(bench_size(basalt, ntr, ns, args, ncpu))

    write_csv(args.csv_out, rows)
    print()
    print(f"{'size':>11} {'impl':<21} {'setting':<26} {'median_ms':>10} {'p90_ms':>9} "
          f"{'gathers/s':>10} {'relRMS vs basalt t=1':>21}")
    for r in rows:
        err = r["relrms_vs_basalt_t1"]
        err_text = "n/a (no mask)" if err is None else f"{err:.3e}"
        print(f"{r['ntraces']}x{r['nsamples']:<6} {r['impl']:<21} {r['setting']:<26} "
              f"{r['median_ms']:>10.3f} {r['p90_ms']:>9.3f} {r['gathers_per_s']:>10.2f} "
              f"{err_text:>21}")
    print(f"csv: {args.csv_out}")


if __name__ == "__main__":
    main()
