"""Tests for the basalt Python bindings (docs/design/FK_PIPELINE.md, sections 2-4 and 7).

Runs under pytest or directly (`python3 test_basalt.py`). The extension must be importable,
e.g. with PYTHONPATH=<build>/python. The numpy reference below is written from the spec, not
from the C++ source, so it is an independent check of the pipeline.
"""

import math
import os
import pathlib
import struct
import tempfile
import threading
import time
import tracemalloc

import numpy as np

import basalt


# --- helpers -----------------------------------------------------------------

# Message of the binding for NaN or inf input (python/src/basalt_py.cpp).
NON_FINITE_MESSAGE = "fk_filter: data must contain only finite values (no NaN or inf)"

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


def _ref_weight(f_hz, k_cpm, v_min, v_max, taper, reject=True, preserve_dc=True):
    """Spec section 3 mute weight, vectorised over broadcastable f and k arrays.

    The DC bin (f = 0, k = 0) has weight 1 when preserve_dc is set. Otherwise it is evaluated
    at v = 0, the f = 0 rule of spec section 2.
    """
    f = np.asarray(f_hz, dtype=np.float64)
    k = np.asarray(k_cpm, dtype=np.float64)
    f, k = np.broadcast_arrays(f, k)
    k_zero = k == 0.0
    dc = k_zero & (f == 0.0)
    safe_k = np.where(k_zero, 1.0, k)
    v = np.where(k_zero, np.inf, -f / safe_k)          # vertical events: v = +inf
    v = np.where(dc, 0.0, v)
    h_min = 0.5 * taper * abs(v_min)
    h_max = 0.5 * taper * abs(v_max)
    inside = _rise(v, v_min, h_min) * (1.0 - _rise(v, v_max, h_max))
    weight = 1.0 - inside if reject else inside
    if preserve_dc:
        weight = np.where(dc, 1.0, weight)
    return weight


def _reference_fk_filter(gather, dt, dx, v_min, v_max, taper, reject=True, pad_factor=1.5,
                         edge_taper_traces=4, edge_taper_samples=0, preserve_dc=True):
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
    weights = _ref_weight(f[None, :], k[:, None], v_min, v_max, taper, reject, preserve_dc)
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


_RAW_SEGY_DTYPES = {3: ">i2", 2: ">i4", 8: ">i1"}   # SEG-Y format codes for int16, int32, int8


def _write_raw_segy(path, values, format_code, dt_us):
    """Writes a big-endian SEG-Y file with integer samples (the writer only does IBM and IEEE).

    `values` is an integer array of shape (ntraces, nsamples). The textual header is blank, the
    extended header count is 0 and each trace header carries its sample count.
    """
    ntraces, nsamples = values.shape
    binary = bytearray(400)
    struct.pack_into(">H", binary, 16, dt_us)        # bytes 3217-3218: sample interval (us)
    struct.pack_into(">H", binary, 20, nsamples)     # bytes 3221-3222: samples per trace
    struct.pack_into(">H", binary, 24, format_code)  # bytes 3225-3226: sample format code
    payload = values.astype(_RAW_SEGY_DTYPES[format_code])
    body = bytearray()
    for t in range(ntraces):
        header = bytearray(240)
        struct.pack_into(">H", header, 114, nsamples)  # bytes 115-116: trace sample count
        body += header
        body += payload[t].tobytes()
    with open(path, "wb") as fh:
        fh.write(b" " * 3200 + bytes(binary) + bytes(body))


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


def test_non_string_mode_and_format_raise_value_error():
    gather = np.zeros((4, 8), dtype=np.float32)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, mode=None)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, mode=1)
    _raises(ValueError, basalt.fk_mute_weight, 1.0, 0.1, mode=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad_format.sgy")
        _raises(ValueError, basalt.write_segy, path, gather, 0.004, format=None)
        _raises(ValueError, basalt.write_segy, path, gather, 0.004, format=b"ieee")
        assert not os.path.exists(path)


def test_non_integer_counts_raise_value_error():
    gather = np.ones((4, 8), dtype=np.float32)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, threads=1.5)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, threads=2 ** 70)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, threads=True)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, threads="2")
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, edge_taper_samples=0.5)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, edge_taper_traces=False)
    # Any integer type with __index__ is accepted, including numpy integers.
    out = basalt.fk_filter(gather, 0.004, 25.0, threads=np.int64(2), edge_taper_traces=np.uint8(2))
    assert out.shape == gather.shape


def test_counts_up_to_2_pow_64_minus_1_are_accepted():
    gather = np.ones((4, 8), dtype=np.float32)
    # Edge tapers wider than the gather are clamped to its length, so huge counts are valid.
    for kwargs in (dict(edge_taper_traces=2 ** 63), dict(edge_taper_samples=2 ** 64 - 1)):
        out = basalt.fk_filter(gather, 0.004, 25.0, **kwargs)
        assert out.shape == gather.shape and np.all(np.isfinite(out)), kwargs
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, edge_taper_samples=2 ** 64)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, edge_taper_traces=-2 ** 63)


def test_unconvertible_gather_raises_value_error():
    _raises(ValueError, basalt.fk_filter, [[1.0, 2.0], [3.0]], 0.004, 25.0)   # ragged list
    _raises(ValueError, basalt.fk_filter, "abcd", 0.004, 25.0)                # string
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ragged.sgy")
        _raises(ValueError, basalt.write_segy, path, [[1.0, 2.0], [3.0]], 0.004)
        _raises(ValueError, basalt.write_segy, path, "abcd", 0.004)
        assert not os.path.exists(path)


def test_list_gather_matches_array_gather():
    rng = np.random.default_rng(10)
    gather = rng.standard_normal((6, 9)).astype(np.float32)
    from_array = basalt.fk_filter(gather, 0.004, 25.0, v_min=-800.0, v_max=1200.0, taper=0.2)
    from_list = basalt.fk_filter(gather.tolist(), 0.004, 25.0, v_min=-800.0, v_max=1200.0,
                                 taper=0.2)
    assert np.array_equal(from_list, from_array)


def test_os_pathlike_paths_are_accepted():
    rng = np.random.default_rng(11)
    samples = rng.standard_normal((3, 16)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "pathlike.sgy"
        basalt.write_segy(path, samples, 0.004)
        assert path.exists()
        got = basalt.read_segy(path)
        assert np.array_equal(got["samples"], samples)
        got_bytes = basalt.read_segy(os.fsencode(str(path)))
        assert np.array_equal(got_bytes["samples"], samples)
        # Non-path objects follow os.fspath and raise TypeError.
        _raises(TypeError, basalt.read_segy, 42)


def test_preserve_dc_false_evaluates_dc_bin_at_zero_velocity():
    rng = np.random.default_rng(9)
    # A large mean puts most of the energy in the DC bin.
    gather = (rng.standard_normal((37, 53)) + 5.0).astype(np.float32)
    params = dict(v_min=-100.0, v_max=100.0, taper=0.0, edge_taper_traces=3,
                  edge_taper_samples=2)
    # The reject band [-100, 100] contains v = 0, so with preserve_dc off the DC bin is muted.
    kept = basalt.fk_filter(gather, 0.004, 25.0, mode="reject", preserve_dc=True, **params)
    removed = basalt.fk_filter(gather, 0.004, 25.0, mode="reject", preserve_dc=False, **params)
    expected_kept = _reference_fk_filter(gather, 0.004, 25.0, reject=True, preserve_dc=True,
                                         **params)
    expected_removed = _reference_fk_filter(gather, 0.004, 25.0, reject=True, preserve_dc=False,
                                            **params)
    assert _relrms(kept, expected_kept) < 1e-4, _relrms(kept, expected_kept)
    assert _relrms(removed, expected_removed) < 1e-4, _relrms(removed, expected_removed)
    assert _relrms(kept, removed) > 0.1, _relrms(kept, removed)


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


def test_tiny_gathers_match_reference():
    rng = np.random.default_rng(12)
    for shape in ((1, 1), (1, 9), (9, 1)):
        gather = rng.standard_normal(shape).astype(np.float32)
        # Identity band without edge tapers returns the input.
        out = basalt.fk_filter(gather, 0.004, 25.0, v_min=1e29, v_max=1e30, taper=0.0,
                               edge_taper_traces=0, edge_taper_samples=0)
        assert _relrms(out, gather) < 1e-5, (shape, _relrms(out, gather))
        # A reject band without edge tapers matches the reference.
        params = dict(v_min=300.0, v_max=900.0, taper=0.1, edge_taper_traces=0,
                      edge_taper_samples=0)
        expected = _reference_fk_filter(gather, 0.004, 25.0, reject=True, **params)
        out = basalt.fk_filter(gather, 0.004, 25.0, **params)
        assert _relrms(out, expected) < 1e-4, (shape, _relrms(out, expected))
        # Default edge tapers wider than a short gather are accepted; the result stays finite.
        assert np.all(np.isfinite(basalt.fk_filter(gather, 0.004, 25.0))), shape


def test_spec_size_1001_by_1500_matches_reference():
    rng = np.random.default_rng(13)
    gather = rng.standard_normal((1001, 1500)).astype(np.float32)
    params = dict(v_min=-800.0, v_max=1200.0, taper=0.1, edge_taper_traces=4,
                  edge_taper_samples=0)
    expected = _reference_fk_filter(gather, 0.004, 25.0, reject=True, **params)
    out = basalt.fk_filter(gather, 0.004, 25.0, mode="reject", **params)
    assert out.shape == (1001, 1500)
    assert _relrms(out, expected) < 1e-4, _relrms(out, expected)


def test_pass_mode_huge_band_removes_trace_mean_but_keeps_dc():
    rng = np.random.default_rng(14)
    nt, ns = 37, 53
    gather = rng.standard_normal((nt, ns)).astype(np.float32)
    params = dict(v_min=-1e30, v_max=1e30, taper=0.0, edge_taper_traces=0,
                  edge_taper_samples=0)
    out = basalt.fk_filter(gather, 0.004, 25.0, mode="pass", **params)
    # Every finite velocity is inside the band and the k = 0 row (v = +inf) is outside it, so
    # only the DC bin of that row survives. On the padded grid that removes the per-sample
    # trace mean and keeps the grand mean (spec section 2 and section 4 note).
    ntr_pad = _smooth_at_least(math.ceil(nt * 1.5), (2, 3, 5, 7, 11))
    ns_pad = _smooth_at_least(math.ceil(ns * 1.5), (2, 3, 5))
    padded = np.zeros((ntr_pad, ns_pad))
    padded[:nt, :ns] = gather
    trace_mean = padded.sum(axis=0) / ntr_pad
    grand_mean = padded.sum() / (ntr_pad * ns_pad)
    expected = (padded - (trace_mean - grand_mean)[None, :])[:nt, :ns]
    assert _relrms(out, expected) < 1e-4, _relrms(out, expected)
    reference = _reference_fk_filter(gather, 0.004, 25.0, reject=False, **params)
    assert _relrms(out, reference) < 1e-4, _relrms(out, reference)


def test_segy_integer_formats_decode():
    dt_us = 4000
    cases = {
        "int8": (8, np.array([[-128, -1, 0, 1, 127, 42], [5, -5, 100, -100, 64, 3]])),
        "int16": (3, np.array([[-32768, -1, 0, 1, 32767, 1024],
                               [7, -7, 300, -300, 16384, 2]])),
        "int32": (2, np.array([[-(2 ** 30), -1, 0, 1, 2 ** 30, 3 * 2 ** 20],
                               [9, -9, 2 ** 24, -2 ** 24, 65536, 4]])),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, (code, values) in cases.items():
            path = os.path.join(tmp, name + ".sgy")
            _write_raw_segy(path, values, code, dt_us)
            got = basalt.read_segy(path)
            assert got["format"] == name, (name, got["format"])
            assert got["ntraces"] == 2 and got["nsamples"] == 6, name
            assert abs(got["dt"] - 0.004) < 1e-12, name
            assert got["samples"].dtype == np.float32, name
            assert np.array_equal(got["samples"], values.astype(np.float32)), name


def test_embedded_nul_in_path_raises_value_error():
    samples = np.ones((2, 8), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "nul\x00name.sgy")
        _raises(ValueError, basalt.write_segy, bad, samples, 0.004)
        _raises(ValueError, basalt.write_segy, os.fsencode(bad), samples, 0.004)
        _raises(ValueError, basalt.read_segy, bad)
        # Nothing is created: the name is not truncated to "nul" in the target directory.
        assert os.listdir(tmp) == []


def test_non_finite_gather_raises_value_error():
    gather = np.ones((4, 8), dtype=np.float32)
    for bad in (np.nan, np.inf, -np.inf):
        corrupted = gather.copy()
        corrupted[2, 3] = bad
        _raises(ValueError, basalt.fk_filter, corrupted, 0.004, 25.0)
        _raises(ValueError, basalt.fk_filter, corrupted, 0.004, 25.0, threads=0)
    # The binding reports non-finite input with its own message, which names the argument.
    try:
        basalt.fk_filter(corrupted, 0.004, 25.0)
    except ValueError as err:
        assert str(err) == NON_FINITE_MESSAGE, str(err)
    else:
        raise AssertionError("expected ValueError for a non-finite sample")


def test_non_finite_sample_is_found_in_large_gather_for_any_thread_count():
    rng = np.random.default_rng(31)
    gather = rng.uniform(-1.0, 1.0, (300, 400)).astype(np.float32)
    for row, col in ((0, 0), (299, 399), (150, 200)):
        for bad in (np.nan, np.inf):
            corrupted = gather.copy()
            corrupted[row, col] = bad
            for threads in (1, 2, 0):
                _raises(ValueError, basalt.fk_filter, corrupted, 0.004, 25.0,
                        threads=threads)
    # A float64 gather with a NaN is rejected too, after the conversion to float32.
    as_double = gather.astype(np.float64)
    as_double[1, 1] = np.nan
    _raises(ValueError, basalt.fk_filter, as_double, 0.004, 25.0)


def test_read_only_float32_gather_is_accepted_and_not_modified():
    rng = np.random.default_rng(33)
    gather = rng.uniform(-1.0, 1.0, (45, 77)).astype(np.float32)
    expected = basalt.fk_filter(gather, 0.004, 25.0, v_min=-800.0, v_max=1200.0)
    readonly = gather.copy()
    readonly.flags.writeable = False
    snapshot = readonly.copy()
    result = basalt.fk_filter(readonly, 0.004, 25.0, v_min=-800.0, v_max=1200.0)
    assert np.array_equal(result, expected)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    # The input is only read: its values and its read-only flag are the same after the call.
    assert np.array_equal(readonly, snapshot)
    assert not readonly.flags.writeable


def test_c_contiguous_float32_input_is_not_copied():
    # numpy reports its buffers to tracemalloc, so the peak traced during a call is the output plus
    # any numpy-side copy of the input. The C++ core's own work buffers, and the copies the binding
    # makes in C++ (Fortran-order or strided input), are not traced. Big-endian float32 is the
    # control: numpy converts it to native float32 before the call, and that copy must show in the
    # trace, or the measurement is void.
    rng = np.random.default_rng(34)
    c_gather = rng.standard_normal((1024, 1024)).astype(np.float32)
    be_gather = c_gather.astype(">f4")
    nbytes = c_gather.nbytes

    def peak_during_filter(gather):
        tracemalloc.start()
        try:
            basalt.fk_filter(gather, 0.004, 25.0, v_min=-800.0, v_max=1200.0)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    c_peak = peak_during_filter(c_gather)
    be_peak = peak_during_filter(be_gather)
    assert c_peak < 1.5 * nbytes, (c_peak, nbytes)
    assert be_peak > 1.8 * nbytes, (be_peak, nbytes)


def test_strided_and_float64_inputs_match_contiguous_float32():
    rng = np.random.default_rng(36)
    wide = rng.uniform(-1.0, 1.0, (90, 160))
    params = dict(v_min=-800.0, v_max=1200.0, taper=0.1, edge_taper_traces=4)
    cases = {
        "float32 column slice": wide.astype(np.float32)[:, ::2],
        "float32 both strided": wide.astype(np.float32)[::3, 1::2],
        "float64 row slice": wide[::2, :],
        "float64 Fortran": np.asfortranarray(wide),
        "float64 transposed view": wide.T,
    }
    for name, view in cases.items():
        expected = basalt.fk_filter(np.ascontiguousarray(view, dtype=np.float32), 0.004, 25.0,
                                    **params)
        got = basalt.fk_filter(view, 0.004, 25.0, **params)
        assert got.dtype == np.float32 and got.flags.c_contiguous, name
        assert np.array_equal(got, expected), name


def test_big_endian_input_matches_native_input():
    rng = np.random.default_rng(37)
    native = rng.uniform(-1.0, 1.0, (33, 70)).astype(np.float32)
    params = dict(v_min=-800.0, v_max=1200.0)
    expected = basalt.fk_filter(native, 0.004, 25.0, **params)
    for big in (native.astype(">f4"), native.astype(">f8")):
        got = basalt.fk_filter(big, 0.004, 25.0, **params)
        assert np.array_equal(got, expected), big.dtype


def test_float64_rounding_to_float32_matches_numpy():
    # Values at and just beyond the float32 rounding boundary, plus non-finite and out-of-range
    # values, stored in strided float64 memory so the GIL-free conversion is the one under test.
    flt_max = float(np.finfo(np.float32).max)
    boundary = math.ldexp(1.0, 128) - math.ldexp(1.0, 103)   # rounds to infinity in float32
    below = math.nextafter(boundary, 0.0)
    row = [flt_max, below, boundary, math.nextafter(boundary, math.inf), -boundary, -below,
           1e300, -1e300, math.inf, -math.inf, math.nan, 1.5, 0.1, -0.0]
    values = np.array([row, row[::-1]], dtype=np.float64)
    wide = np.zeros((2, 2 * values.shape[1]))
    wide[:, ::2] = values
    strided = wide[:, ::2]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "round.sgy")
        basalt.write_segy(path, strided, 0.004, format="ieee")
        got = basalt.read_segy(path)["samples"]
    with np.errstate(over="ignore", invalid="ignore"):
        expected = strided.astype(np.float32)
    np.testing.assert_array_equal(got, expected)
    assert got[0, 1] == flt_max and np.isinf(got[0, 2]), got[0, :3]


def test_strided_float64_out_of_range_and_non_finite_are_reported():
    wide = np.zeros((4, 32))
    view = wide[:, ::2]
    view[:] = 1.0
    view[1, 1] = 1e300
    with np.errstate(over="ignore"):
        try:
            basalt.fk_filter(view, 0.004, 25.0)
        except ValueError as err:
            message = str(err)
        else:
            raise AssertionError("expected ValueError for values beyond the float32 range")
    assert "float32" in message and "NaN" not in message, message
    view[1, 1] = np.nan
    try:
        basalt.fk_filter(view, 0.004, 25.0)
    except ValueError as err:
        assert str(err) == NON_FINITE_MESSAGE, str(err)
    else:
        raise AssertionError("expected ValueError for a NaN sample")


def _spinner_ratio(busy_call):
    """Throughput of a spinning thread while busy_call runs, relative to its idle rate."""
    counter = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter[0] += 1

    ticker = threading.Thread(target=spin)
    ticker.start()
    try:
        time.sleep(0.05)
        t0, c0 = time.perf_counter(), counter[0]
        time.sleep(0.3)
        idle_rate = (counter[0] - c0) / (time.perf_counter() - t0)
        t0, c0 = time.perf_counter(), counter[0]
        while True:
            busy_call()
            if time.perf_counter() - t0 >= 0.3:
                break
        busy_rate = (counter[0] - c0) / (time.perf_counter() - t0)
    finally:
        stop.set()
        ticker.join()
    assert idle_rate > 0, idle_rate
    return busy_rate / idle_rate


def test_gil_released_during_segy_io():
    # write_segy and read_segy release the GIL for their whole call, including the conversion of a
    # float64 Fortran-order gather. This is a coarse guard: it fails only if the GIL is held for
    # most of a call, which the conversion alone would not cause.
    rng = np.random.default_rng(35)
    gather = np.asfortranarray(rng.standard_normal((2000, 2000)))   # float64, Fortran order
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gil.sgy")
        basalt.write_segy(path, gather, 0.004)
        write_ratio = _spinner_ratio(lambda: basalt.write_segy(path, gather, 0.004))
        read_ratio = _spinner_ratio(lambda: basalt.read_segy(path))
    assert write_ratio > 0.25, f"spinner ran at {write_ratio:.3f} of its idle rate in write_segy"
    assert read_ratio > 0.25, f"spinner ran at {read_ratio:.3f} of its idle rate in read_segy"


def test_non_finite_frequency_or_wavenumber_raises_value_error():
    for f, k in ((math.nan, 0.01), (math.inf, 0.01), (10.0, math.nan), (10.0, -math.inf)):
        _raises(ValueError, basalt.fk_mute_weight, f, k, v_min=500.0, v_max=2000.0)


def test_step_overflow_and_output_overflow_raise_value_error():
    gather = np.ones((37, 53), dtype=np.float32)
    # 1 / (ns_pad * dt) and 1 / (ntr_pad * dx) overflow to inf for subnormal steps.
    _raises(ValueError, basalt.fk_filter, gather, 1e-320, 25.0, v_min=-800.0, v_max=1200.0)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 1e-320, v_min=-800.0, v_max=1200.0)
    # Finite input whose transform overflows float32 is reported, not returned as NaN.
    rng = np.random.default_rng(15)
    big = (rng.standard_normal((37, 53)) * 1e36).astype(np.float32)
    _raises(ValueError, basalt.fk_filter, big, 0.004, 25.0, v_min=-800.0, v_max=1200.0)


def test_identity_configuration_overflow_is_reported_not_returned():
    # Finite float32 input at 3e38: the transform overflows even though every weight is 1.
    gather = np.full((37, 53), 3e38, dtype=np.float32)
    identity = dict(v_min=1e29, v_max=1e30, taper=0.0, edge_taper_traces=0, edge_taper_samples=0)
    _raises(ValueError, basalt.fk_filter, gather, 0.004, 25.0, **identity)


def test_finite_float64_beyond_float32_range_is_reported_as_range_error():
    # 1e300 is finite in float64; the float32 conversion makes it infinite. The message must not
    # claim that the data contains NaN or inf. A real NaN in the same input still gets that message.
    gather = np.full((4, 8), 1e300)
    # numpy warns about the float64 to float32 overflow in the conversion; the binding reports it.
    with np.errstate(over="ignore"):
        try:
            basalt.fk_filter(gather, 0.004, 25.0)
        except ValueError as err:
            message = str(err)
        else:
            raise AssertionError("expected ValueError for values beyond the float32 range")
        assert "float32" in message and "NaN" not in message, message

        gather[1, 1] = np.nan
        try:
            basalt.fk_filter(gather, 0.004, 25.0)
        except ValueError as err:
            message = str(err)
        else:
            raise AssertionError("expected ValueError for a NaN sample")
        assert message == NON_FINITE_MESSAGE, message


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
