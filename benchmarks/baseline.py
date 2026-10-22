#!/usr/bin/env python3
"""NumPy reference for the Basalt F-K pipeline.

Each timed shot runs the same work as BatchExecutor::process_job:
forward 2D FFT, F-K band-mute filter, inverse 2D FFT. Everything is
complex64/float32, the filter uses the same velocity axis and sigmoid band
weight as src/filter/fk_filter_impl.hpp, and the filter mask is built once
outside the timed loop.
"""

import argparse
import sys
import time

import numpy as np

# Defaults match benchmarks/basalt_bench.cpp.
DT = 0.004          # sample interval (s)
DX = 5.0            # trace interval (m)
MUTE_VEL_MIN = -1500.0
MUTE_VEL_MAX = 1500.0
TAPER_WIDTH = 200.0  # sigmoid taper width (m/s)

SIGMOID_CLAMP = 87.0  # same input clamp as basalt::utils::fast_sigmoid_avx


def wrapped_axis(n, step):
    """Frequency/wavenumber axis in Basalt order: indices <= n/2 are positive."""
    idx = np.arange(n)
    signed = np.where(idx <= n // 2, idx, idx - n)
    return (signed * step).astype(np.float32)


def sigmoid(x):
    x = np.clip(x, -SIGMOID_CLAMP, SIGMOID_CLAMP)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x))


def build_mask(rows, cols):
    """Band-mute weight on the (rows, cols) f-k grid, float32.

    v = f / k, weight = 1 - sigmoid((v - vmin) * s) * sigmoid((vmax - v) * s),
    with s = 10 / taper_width. Where k == 0 the weight is 1 (pass).
    """
    df = 1.0 / (rows * DT)
    dk = 1.0 / (cols * DX)
    freqs = wrapped_axis(rows, df)
    knums = wrapped_axis(cols, dk)
    K, F = np.meshgrid(knums, freqs)  # shape (rows, cols)

    nonzero_k = K != 0
    safe_k = np.where(nonzero_k, K, np.float32(1.0)).astype(np.float32)
    V = (F / safe_k).astype(np.float32)

    steepness = np.float32(10.0 / TAPER_WIDTH)
    lo = sigmoid((V - np.float32(MUTE_VEL_MIN)) * steepness)
    hi = sigmoid((np.float32(MUTE_VEL_MAX) - V) * steepness)
    weight = np.float32(1.0) - lo * hi
    return np.where(nonzero_k, weight, np.float32(1.0)).astype(np.float32)


def make_input(n, seed):
    rng = np.random.default_rng(seed)
    real = rng.uniform(-1.0, 1.0, size=(n, n))
    imag = rng.uniform(-1.0, 1.0, size=(n, n))
    return (real + 1j * imag).astype(np.complex64)


def run_pipeline(data, mask):
    spectrum = np.fft.fft2(data)
    filtered = spectrum * mask
    return np.fft.ifft2(filtered)


def benchmark_fk_filter(size, iters, warmup, seed):
    data = make_input(size, seed)
    mask = build_mask(size, size)

    # Dtype check: NumPy must keep single precision end to end.
    probe = run_pipeline(data, mask)
    if probe.dtype != np.complex64:
        raise SystemExit(
            f"expected complex64 pipeline output, got {probe.dtype}; "
            "this benchmark needs NumPy >= 2.0"
        )

    for _ in range(warmup):
        run_pipeline(data, mask)

    times = []
    result = None
    for _ in range(iters):
        t0 = time.perf_counter()
        result = run_pipeline(data, mask)
        times.append(time.perf_counter() - t0)

    if not np.isfinite(result).all():
        raise SystemExit(f"size {size}: non-finite output, timings are not valid")

    mean_s = sum(times) / len(times)
    return mean_s, min(times)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="NumPy baseline for the Basalt F-K pipeline (fft2 -> band mute -> ifft2)."
    )
    parser.add_argument(
        "--size", type=int, nargs="+", default=[256, 512, 1024, 2048],
        help="grid sizes N (NxN, power of two). Default: 256 512 1024 2048",
    )
    parser.add_argument("--iters", type=int, default=20, help="timed iterations per size. Default: 20")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations per size. Default: 5")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for the input. Default: 1337")
    args = parser.parse_args(argv)

    for n in args.size:
        if n <= 0 or (n & (n - 1)) != 0:
            parser.error(f"--size values must be powers of two, got {n}")
    if args.iters < 1:
        parser.error("--iters must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(
        f"Benchmarking Python/NumPy baseline (complex64, dt={DT} dx={DX} "
        f"band=[{MUTE_VEL_MIN:g}, {MUTE_VEL_MAX:g}] taper={TAPER_WIDTH:g}, "
        f"iters={args.iters}, warmup={args.warmup}, numpy {np.__version__})"
    )
    for size in args.size:
        mean_s, best_s = benchmark_fk_filter(size, args.iters, args.warmup, args.seed)
        print(f"Size {size}x{size}: {mean_s * 1000:.2f} ms mean, {best_s * 1000:.2f} ms best")
    return 0


if __name__ == "__main__":
    sys.exit(main())
