"""Tests for the basalt Python bindings (docs/design/FK_PIPELINE.md, sections 2-4 and 7).

Runs under pytest or directly (`python3 test_basalt.py`). The extension must be importable,
e.g. with PYTHONPATH=<build>/python. The numpy reference below is written from the spec, not
from the C++ source, so it is an independent check of the pipeline.
"""

import math
import os
import tempfile
import threading
import time

import numpy as np

import basalt


# --- helpers -----------------------------------------------------------------

def _relrms(actual, expected):
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.sqrt(np.mean((np.asarray(actual, dtype=np.float64) - expected) ** 2))
                 / np.sqrt(np.mean(expected ** 2)))


def _raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    except Exception as err:  # wrong exception type
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(err).__name__}: {err}") from err
    raise AssertionError(f"expected {exc_type.__name__}, but nothing was raised")


def _smooth_at_least(n, primes):
    """Smallest integer >= n whose prime factors all lie in `primes` (pocketfft good sizes)."""
    m = max(n, 1)
    while True:
        x = m
        for p in primes:
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1


def _edge_weights(n, width):
    """Hann half-window edge taper on both ends of a line of length n (spec section 4, step 3)."""
    w = np.ones(n)
    m = min(width, n)
    i = np.arange(m)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (width + 1.0))
    w[:m] *= ramp
    w[n - 1 - i] *= ramp
    return w


def _rise(v, edge, half):
    """Raised-cosine step from 0 to 1 centred on `edge` (spec section 3)."""
    lo = edge - half
    hi = edge + half
    if half == 0.0:
        return np.where(v <= edge, 0.0, 1.0)
    with np.errstate(all="ignore"):
        ramp = 0.5 - 0.5 * np.cos(np.pi * (v - lo) / (2.0 * half))
    return np.where(v <= lo, 0.0, np.where(v >= hi, 1.0, ramp))


def _ref_weight(f_hz, k_cpm, v_min, v_max, taper, reject=True):
    """Spec section 3 mute weight, vectorised over broadcastable f and k arrays (DC preserved)."""
    f = np.asarray(f_hz, dtype=np.float64)
    k = np.asarray(k_cpm, dtype=np.float64)
    f, k = np.broadcast_arrays(f, k)
    k_zero = k == 0.0
    safe_k = np.where(k_zero, 1.0, k)
    v = np.where(k_zero, np.inf, -f / safe_k)          # vertical events: v = +inf
    h_min = 0.5 * taper * abs(v_min)
    h_max = 0.5 * taper * abs(v_max)
    inside = _rise(v, v_min, h_min) * (1.0 - _rise(v, v_max, h_max))
    weight = 1.0 - inside if reject else inside
    dc = k_zero & (f == 0.0)
    return np.where(dc, 1.0, weight)


def _reference_fk_filter(gather, dt, dx, v_min, v_max, taper, reject=True, pad_factor=1.5,
                         edge_taper_traces=4, edge_taper_samples=0):
    """Independent numpy implementation of spec sections 3-4 (double precision)."""
    x = np.asarray(gather, dtype=np.float64)
    nt, ns = x.shape
    ntr_pad = _smooth_at_least(math.ceil(nt * pad_factor), (2, 3, 5, 7, 11))
    ns_pad = _smooth_at_least(math.ceil(ns * pad_factor), (2, 3, 5))
    padded = np.zeros((ntr_pad, ns_pad))
    padded[:nt, :ns] = (x * _edge_weights(nt, edge_taper_traces)[:, None]
                        * _edge_weights(ns, edge_taper_samples)[None, :])
    spectrum = np.fft.rfft2(padded)
    f = np.arange(ns_pad // 2 + 1) / (ns_pad * dt)
    k = np.fft.fftfreq(ntr_pad, d=dx)
    weights = _ref_weight(f[None, :], k[:, None], v_min, v_max, taper, reject)
    filtered = np.fft.irfft2(spectrum * weights, s=(ntr_pad, ns_pad))
    return filtered[:nt, :ns]


def _dipping_event(nt, ns, dt, dx, velocity, t_center, wavelet_hz=25.0):
    """Ricker wavelet on a line with arrival t(x) = t_center + x / velocity, Hann-tapered in x."""
    x = (np.arange(nt) - (nt - 1) / 2.0) * dx
    t = np.arange(ns) * dt
    envelope = np.hanning(nt)
    arrival = t_center + x / velocity
    tau = t[None, :] - arrival[:, None]
    a = np.pi * wavelet_hz
    ricker = (1.0 - 2.0 * (a * tau) ** 2) * np.exp(-((a * tau) ** 2))
    return (envelope[:, None] * ricker).astype(np.float32)


# --- tests -------------------------------------------------------------------

def test_package_exports():
    assert basalt.__version__ == "0.2.0"
    for name in ("fk_filter", "fk_mute_weight", "read_segy", "write_segy"):
        assert callable(getattr(basalt, name)), name


def test_identity_configuration_returns_input():
    rng = np.random.default_rng(0)
    gather = rng.standard_normal((37, 53)).astype(np.float32)
    out = basalt.fk_filter(gather, 0.004, 25.0, v_min=1e29, v_max=1e30, taper=0.0,
                           mode="reject", edge_taper_traces=0, edge_taper_samples=0)
    assert out.dtype == np.float32
    assert out.shape == gather.shape
    assert out.flags.c_contiguous
    assert not np.shares_memory(out, gather)
    assert _relrms(out, gather) < 1e-5


def test_matches_numpy_reference_asymmetric_band_and_tapers():
    rng = np.random.default_rng(1)
    gather = rng.standard_normal((37, 53)).astype(np.float32)
    params = dict(v_min=-800.0, v_max=1200.0, taper=0.2, pad_factor=1.5,
                  edge_taper_traces=3, edge_taper_samples=7)
    expected = _reference_fk_filter(gather, 0.004, 25.0, reject=True, **params)
    out = basalt.fk_filter(gather, 0.004, 25.0, mode="reject", **params)
    assert _relrms(out, expected) < 1e-4, _relrms(out, expected)
    # The filter must actually change the data for this band.
    assert _relrms(out, gather) > 0.1


def test_matches_numpy_reference_pass_mode_and_small_grids():
    rng = np.random.default_rng(2)
    cases = [
        ((37, 53), dict(v_min=-800.0, v_max=1200.0, taper=0.2, edge_taper_traces=3,
                        edge_taper_samples=7), 1.5, "Pass", False),
        ((7, 13), dict(v_min=300.0, v_max=900.0, taper=0.1, edge_taper_traces=2,
                       edge_taper_samples=2), 1.0, "reject", True),
    ]
    for shape, params, pad, mode, reject in cases:
        gather = rng.standard_normal(shape).astype(np.float32)
        expected = _reference_fk_filter(gather, 0.002, 10.0, reject=reject, pad_factor=pad,
                                        **params)
        out = basalt.fk_filter(gather, 0.002, 10.0, mode=mode, pad_factor=pad, **params)
        assert _relrms(out, expected) < 1e-4, (shape, mode, _relrms(out, expected))


def test_dipping_event_attenuation_depends_on_sign():
    nt, ns, dt, dx = 64, 500, 0.002, 5.0
    lo, hi = nt // 4, 3 * nt // 4   # central 50% of traces
    kwargs = dict(v_min=500.0, v_max=2000.0, taper=0.1, mode="reject")

    positive = _dipping_event(nt, ns, dt, dx, +1000.0, t_center=0.5)
    out_pos = basalt.fk_filter(positive, dt, dx, **kwargs)
    att_pos_db = 20.0 * math.log10(
        np.sqrt(np.mean(out_pos[lo:hi] ** 2.0)) / np.sqrt(np.mean(positive[lo:hi] ** 2.0)))
    assert att_pos_db <= -20.0, att_pos_db

    negative = _dipping_event(nt, ns, dt, dx, -1000.0, t_center=0.5)
    out_neg = basalt.fk_filter(negative, dt, dx, **kwargs)
    loss_neg_db = 20.0 * math.log10(
        np.sqrt(np.mean(out_neg[lo:hi] ** 2.0)) / np.sqrt(np.mean(negative[lo:hi] ** 2.0)))
    assert loss_neg_db > -1.0, loss_neg_db


def test_fk_mute_weight_values():
    # Reject a band [500, 2000] m/s; v = -f / k.
    assert basalt.fk_mute_weight(10.0, -0.01, v_min=500.0, v_max=2000.0, taper=0.0,
                                 mode="reject") == 0.0      # v = +1000, inside
    assert basalt.fk_mute_weight(10.0, 0.01, v_min=500.0, v_max=2000.0, taper=0.0,
                                 mode="reject") == 1.0      # v = -1000, outside
    assert basalt.fk_mute_weight(10.0, -0.01, v_min=500.0, v_max=2000.0, taper=0.0,
                                 mode="pass") == 1.0
    # Vertical event (k = 0, f != 0): v = +inf, outside every finite band.
    assert basalt.fk_mute_weight(10.0, 0.0, v_min=500.0, v_max=2000.0, taper=0.1,
                                 mode="reject") == 1.0
    assert basalt.fk_mute_weight(10.0, 0.0, v_min=500.0, v_max=2000.0, taper=0.1,
                                 mode="pass") == 0.0
    # DC bin is kept at 1 by default.
    assert basalt.fk_mute_weight(0.0, 0.0, v_min=500.0, v_max=2000.0, mode="pass") == 1.0
    # Flat-in-f event at v = 0 (f = 0, k != 0), band straddling zero.
    assert basalt.fk_mute_weight(0.0, 0.05, v_min=-100.0, v_max=100.0, taper=0.0,
                                 mode="pass") == 1.0
    # Raised-cosine midpoint: edge 500, taper 0.2 gives half width 50; v = 500 sits at weight 0.5.
    w = basalt.fk_mute_weight(5.0, -0.01, v_min=500.0, v_max=2000.0, taper=0.2, mode="reject")
    assert abs(w - 0.5) < 1e-12, w


def test_fk_mute_weight_matches_reference_grid():
    rng = np.random.default_rng(3)
    for _ in range(40):
        f = float(rng.uniform(0.0, 125.0))
        k = float(rng.uniform(-0.02, 0.02))
        v_min, v_max = -700.0, 900.0
        taper = float(rng.uniform(0.0, 0.5))
        for mode, reject in (("reject", True), ("pass", False)):
            got = basalt.fk_mute_weight(f, k, v_min=v_min, v_max=v_max, taper=taper, mode=mode)
            want = float(_ref_weight(f, k, v_min, v_max, taper, reject))
            assert abs(got - want) < 1e-12, (f, k, taper, mode, got, want)


def test_float64_and_fortran_order_inputs_are_accepted():
    rng = np.random.default_rng(4)
    base = rng.standard_normal((37, 53)).astype(np.float32)
    params = dict(v_min=-800.0, v_max=1200.0, taper=0.2, edge_taper_traces=3,
                  edge_taper_samples=7)
    expected = basalt.fk_filter(base, 0.004, 25.0, **params)
    from_f64 = basalt.fk_filter(base.astype(np.float64), 0.004, 25.0, **params)
    from_fortran = basalt.fk_filter(np.asfortranarray(base), 0.004, 25.0, **params)
    assert from_f64.dtype == np.float32 and from_fortran.dtype == np.float32
    assert np.array_equal(from_f64, expected)
    assert np.array_equal(from_fortran, expected)


def test_non_2d_input_raises_value_error():
    _raises(ValueError, basalt.fk_filter, np.zeros(100, dtype=np.float32), 0.004, 25.0)
    _raises(ValueError, basalt.fk_filter, np.zeros((2, 3, 4), dtype=np.float32), 0.004, 25.0)


def test_bad_mode_raises_value_error():
    gather = np.zeros((4, 8), dtype=np.float32)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, mode="notch")
    _raises(ValueError, basalt.fk_mute_weight, 1.0, 0.1, mode="both")
    # Case-insensitive modes are accepted.
    basalt.fk_filter(gather, 0.004, 25.0, mode="PASS")
    basalt.fk_filter(gather, 0.004, 25.0, mode="Reject")


def test_invalid_parameters_raise_value_error():
    gather = np.ones((4, 8), dtype=np.float32)
    _raises(ValueError, basalt.fk_filter, gather, 0.0, 25.0)                 # dt <= 0
    _raises(ValueError, basalt.fk_filter, gather, 0.004, -1.0)               # dx <= 0
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, v_min=5.0, v_max=5.0)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, taper=1.0)    # taper in [0, 1)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, taper=-0.1)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, pad_factor=0.5)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, threads=-1)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, edge_taper_traces=-2)
    _raises(ValueError, basalt.fk_filter, np.zeros((0, 8), dtype=np.float32), 0.004, 25.0)
    _raises(ValueError, basalt.fk_mute_weight, 1.0, 0.1, v_min=2.0, v_max=1.0)
    _raises(ValueError, basalt.fk_mute_weight, 1.0, 0.1, taper=1.5)


def test_threads_argument_gives_same_result():
    rng = np.random.default_rng(5)
    gather = rng.standard_normal((37, 53)).astype(np.float32)
    single = basalt.fk_filter(gather, 0.004, 25.0, v_min=-800.0, v_max=1200.0, taper=0.2,
                              threads=1)
    multi = basalt.fk_filter(gather, 0.004, 25.0, v_min=-800.0, v_max=1200.0, taper=0.2,
                             threads=0)
    assert _relrms(multi, single) < 1e-6


def test_segy_round_trip_ieee_and_ibm():
    rng = np.random.default_rng(6)
    samples = rng.standard_normal((5, 40)).astype(np.float32)
    dt = 0.002
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ieee.sgy")
        basalt.write_segy(path, samples, dt, format="ieee")
        got = basalt.read_segy(path)
        assert got["format"] == "ieee"
        assert got["ntraces"] == 5 and got["nsamples"] == 40
        assert abs(got["dt"] - dt) < 1e-12
        assert isinstance(got["text_header"], str)
        assert got["samples"].dtype == np.float32
        assert got["samples"].shape == (5, 40)
        assert got["samples"].flags.owndata
        assert np.array_equal(got["samples"], samples)

        path_ibm = os.path.join(tmp, "ibm.sgy")
        basalt.write_segy(path_ibm, samples.astype(np.float64), dt, format="IBM")
        got_ibm = basalt.read_segy(path_ibm)
        assert got_ibm["format"] == "ibm"
        assert np.allclose(got_ibm["samples"], samples, rtol=1e-5, atol=1e-6)

        _raises(ValueError, basalt.write_segy, path, samples, dt, format="int16")
        _raises(ValueError, basalt.write_segy, path, samples.ravel(), dt)
        _raises(RuntimeError, basalt.read_segy, os.path.join(tmp, "missing.sgy"))


def test_gil_released_during_filter():
    # A spinning thread needs the GIL for every tick. If fk_filter held the GIL for its whole
    # duration, the spinner's throughput while the filter runs would collapse towards zero (only
    # the interpreter's 5 ms switch interval between calls lets it run); if the GIL is released,
    # it stays near the idle rate (about 1.0 on a multi-core machine, about 0.5 on a single core
    # where the filter and the spinner share the CPU).
    rng = np.random.default_rng(7)
    big = rng.standard_normal((1200, 3000)).astype(np.float32)   # roughly 0.15 s per call
    counter = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter[0] += 1

    ticker = threading.Thread(target=spin)
    ticker.start()
    try:
        # Idle baseline: the main thread sleeps, so the spinner has the interpreter to itself.
        time.sleep(0.05)
        t0, c0 = time.perf_counter(), counter[0]
        time.sleep(0.3)
        idle_rate = (counter[0] - c0) / (time.perf_counter() - t0)

        # Back-to-back filter calls for a measurable window, timed against the same spinner.
        t0, c0 = time.perf_counter(), counter[0]
        while True:
            basalt.fk_filter(big, 0.002, 10.0, v_min=-800.0, v_max=1200.0, taper=0.1)
            if time.perf_counter() - t0 >= 0.3:
                break
        busy_elapsed = time.perf_counter() - t0
        busy_rate = (counter[0] - c0) / busy_elapsed
    finally:
        stop.set()
        ticker.join()
    assert idle_rate > 0, idle_rate
    ratio = busy_rate / idle_rate
    assert ratio > 0.25, f"spinner ran at {ratio:.3f} of its idle rate during fk_filter"


def test_two_threads_filter_concurrently():
    rng = np.random.default_rng(8)
    gather = rng.standard_normal((200, 600)).astype(np.float32)
    params = dict(v_min=-800.0, v_max=1200.0, taper=0.1, edge_taper_traces=4)
    expected = basalt.fk_filter(gather, 0.004, 25.0, **params)
    results = [None, None]
    errors = []

    def worker(slot):
        try:
            for _ in range(3):
                results[slot] = basalt.fk_filter(gather, 0.004, 25.0, **params)
        except Exception as err:  # reported below, not swallowed
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for result in results:
        assert np.array_equal(result, expected)


def _run_all():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, func in tests:
        try:
            func()
        except Exception as err:
            failures += 1
            print(f"FAIL {name}: {err}")
        else:
            print(f"ok   {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
