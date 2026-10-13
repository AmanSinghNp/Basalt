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


# Trace header fields: dtype of the returned array, and the 1-based first SEG-Y byte and struct code ("i" int32, "h" int16) used to write the field.
_HEADER_DTYPES = {
    "trace_sequence": np.int32, "field_record": np.int32, "channel": np.int32, "offset": np.int32,
    "scalar_elev": np.int16, "scalar_coord": np.int16, "coord_units": np.int16,
    "source_x": np.float64, "source_y": np.float64, "group_x": np.float64, "group_y": np.float64,
    "receiver_elevation": np.float64, "delay_ms": np.int16,
    "year": np.int16, "day_of_year": np.int16, "hour": np.int16, "minute": np.int16,
    "second": np.int16,
}
_HEADER_BYTES = {
    "trace_sequence": (5, "i"), "field_record": (9, "i"), "channel": (13, "i"),
    "offset": (37, "i"), "receiver_elevation": (41, "i"),
    "scalar_elev": (69, "h"), "scalar_coord": (71, "h"),
    "source_x": (73, "i"), "source_y": (77, "i"), "group_x": (81, "i"), "group_y": (85, "i"),
    "coord_units": (89, "h"), "delay_ms": (109, "h"),
    "year": (157, "h"), "day_of_year": (159, "h"), "hour": (161, "h"), "minute": (163, "h"),
    "second": (165, "h"),
}
_HEADER_SCALARS = ("scalar_elev", "scalar_coord")


def _write_segy_headers(path, traces, format_code=5, nsamples=4, sample_bytes=None):
    """Writes a big-endian SEG-Y file whose trace headers hold `traces`.

    `traces` is a list of dicts from field name to value; fields that are not named are zero.
    Each trace is followed by `sample_bytes` (zeros in IEEE format by default).
    """
    binary = bytearray(400)
    struct.pack_into(">H", binary, 16, 4000)         # bytes 3217-3218: 4000 us
    struct.pack_into(">H", binary, 20, nsamples)     # bytes 3221-3222
    struct.pack_into(">H", binary, 24, format_code)  # bytes 3225-3226
    if sample_bytes is None:
        sample_bytes = bytes(4 * nsamples)
    body = bytearray()
    for fields in traces:
        header = bytearray(240)
        for name, value in fields.items():
            first, code = _HEADER_BYTES[name]
            struct.pack_into(">" + code, header, first - 1, value)
        struct.pack_into(">H", header, 114, nsamples)  # bytes 115-116: trace sample count
        body += header
        body += sample_bytes
    with open(path, "wb") as fh:
        fh.write(b" " * 3200 + bytes(binary) + bytes(body))


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
    for name in ("fk_filter", "fk_mute_weight", "read_segy", "read_trace_headers", "write_segy"):
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


def test_read_segy_result_has_headers_and_read_trace_headers_has_all_keys():
    rng = np.random.default_rng(21)
    samples = rng.standard_normal((3, 4)).astype(np.float32)
    traces = [{"trace_sequence": 1}, {"trace_sequence": 2}, {"trace_sequence": 3}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "keys.sgy")
        _write_segy_headers(path, traces)
        got = basalt.read_segy(path)
        assert set(got) == {"samples", "dt", "format", "text_header", "ntraces", "nsamples",
                            "headers"}
        headers = basalt.read_trace_headers(path)
        assert set(headers) == set(_HEADER_DTYPES), sorted(set(headers) ^ set(_HEADER_DTYPES))
        assert set(got["headers"]) == set(headers)
        for name, dtype in _HEADER_DTYPES.items():
            assert headers[name].dtype == dtype, (name, headers[name].dtype)
            assert headers[name].shape == (3,), name
            assert got["headers"][name].dtype == dtype, name
            assert np.array_equal(got["headers"][name], headers[name]), name
        assert headers["trace_sequence"].tolist() == [1, 2, 3]
        assert got["samples"].shape == (3, 4)

        path_samples = os.path.join(tmp, "samples.sgy")
        basalt.write_segy(path_samples, samples, 0.004)
        assert basalt.read_segy(path_samples)["ntraces"] == 3


def test_extreme_values_decode_for_each_header_key():
    # Scalars are kept at zero (meaning 1) so every field decodes to its raw value. Trace 0 holds
    # the largest value of each field's type and trace 1 the smallest.
    hi, lo = {}, {}
    for name, (_, code) in _HEADER_BYTES.items():
        if name in _HEADER_SCALARS:
            continue
        hi[name] = 2 ** 31 - 1 if code == "i" else 2 ** 15 - 1
        lo[name] = -2 ** 31 if code == "i" else -2 ** 15
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "extremes.sgy")
        _write_segy_headers(path, [hi, lo])
        headers = basalt.read_trace_headers(path)
        for name, dtype in _HEADER_DTYPES.items():
            if name in _HEADER_SCALARS:
                assert headers[name].tolist() == [0, 0], name
                continue
            expected = np.array([hi[name], lo[name]], dtype=dtype)
            assert np.array_equal(headers[name], expected), (name, headers[name])


def test_positive_negative_and_zero_scalars():
    traces = [
        {"scalar_coord": 10, "source_x": -12, "group_y": 7, "scalar_elev": 2,
         "receiver_elevation": 500},
        {"scalar_coord": -100, "group_x": 12345, "source_y": -1, "scalar_elev": -10,
         "receiver_elevation": -15},
        {"scalar_coord": 0, "group_x": 987654, "source_x": -5, "scalar_elev": 0,
         "receiver_elevation": 42},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scalars.sgy")
        _write_segy_headers(path, traces)
        headers = basalt.read_trace_headers(path)
    # Positive multiplies, negative divides by its absolute value, zero means 1. The scalars
    # themselves come back raw.
    assert headers["scalar_coord"].tolist() == [10, -100, 0]
    assert headers["scalar_elev"].tolist() == [2, -10, 0]
    assert np.array_equal(headers["source_x"], [-120.0, 0.0, -5.0])
    assert np.array_equal(headers["group_x"], [0.0, 123.45, 987654.0])
    assert np.array_equal(headers["group_y"], [70.0, 0.0, 0.0])
    assert np.array_equal(headers["source_y"], [0.0, -0.01, 0.0])
    assert np.array_equal(headers["receiver_elevation"], [1000.0, -1.5, 42.0])


def test_coord_units_values_are_reported_as_stored():
    units = [1, 2, 3, 4, 0, -1]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "units.sgy")
        _write_segy_headers(path, [{"coord_units": u} for u in units])
        headers = basalt.read_trace_headers(path)
    assert headers["coord_units"].dtype == np.int16
    assert headers["coord_units"].tolist() == units


def test_read_trace_headers_does_not_decode_samples():
    # IBM word 0x7F123456 is far beyond the IEEE range. read_segy refuses the file, but
    # read_trace_headers reads only the headers and must succeed.
    sample_bytes = struct.pack(">II", 0x41100000, 0x7F123456)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ibm_headers.sgy")
        _write_segy_headers(path, [{"trace_sequence": 77, "group_y": -600, "scalar_coord": -6}],
                            format_code=1, nsamples=2, sample_bytes=sample_bytes)
        _raises(RuntimeError, basalt.read_segy, path)
        headers = basalt.read_trace_headers(path)
    assert headers["trace_sequence"].tolist() == [77]
    assert headers["group_y"].tolist() == [-100.0]


def test_read_trace_headers_of_file_with_no_traces_and_bad_input():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.sgy")
        _write_segy_headers(path, [])
        headers = basalt.read_trace_headers(path)
        for name, dtype in _HEADER_DTYPES.items():
            assert headers[name].dtype == dtype, name
            assert headers[name].shape == (0,), name

        one = os.path.join(tmp, "one.sgy")
        _write_segy_headers(one, [{"trace_sequence": 5}])
        assert basalt.read_trace_headers(pathlib.Path(one))["trace_sequence"].tolist() == [5]
        _raises(TypeError, basalt.read_trace_headers, 42)
        _raises(RuntimeError, basalt.read_trace_headers, os.path.join(tmp, "missing.sgy"))

        with open(one, "rb") as fh:
            data = fh.read()
        with open(one, "wb") as fh:
            fh.write(data[:-1])  # truncated trace: the layout is still checked
        _raises(RuntimeError, basalt.read_trace_headers, one)


def _trace_header_bytes(path, trace, nsamples, sample_bytes=4):
    """The raw 240 header bytes of one trace in an IEEE or IBM file written by write_segy."""
    with open(path, "rb") as fh:
        data = fh.read()
    start = 3600 + trace * (240 + nsamples * sample_bytes)
    return data[start:start + 240]


def test_write_segy_headers_round_trip_through_read_trace_headers_and_read_segy():
    rng = np.random.default_rng(41)
    samples = rng.standard_normal((4, 5)).astype(np.float32)
    headers = {
        "trace_sequence": [101, 102, 103, 104],
        "field_record": [7001, 7001, 7002, 7002],
        "channel": [1, 2, 1, 2],
        "offset": [-250, 0, 1200, 2_000_000_000],
        "receiver_elevation": [1234.5, -15.25, 0.0, 2000.0],
        "source_x": [-12.5, 0.0, 100.0, 999.75],
        "source_y": [5.0, -0.01, 0.0, 7.0],
        "group_x": [123.4, 4500.0, -987.0, 0.0],
        "group_y": [-3.2, 27.0, 1.0, 12345.0],
        "coord_units": [1, 1, 1, 1],
        "delay_ms": [-20, 0, 15, 300],
        "year": [2026, 2026, 2026, 2026],
        "day_of_year": [286, 286, 287, 287],
        "hour": [23, 0, 12, 6],
        "minute": [59, 1, 30, 45],
        "second": [58, 2, 1, 59],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "written.sgy")
        basalt.write_segy(path, samples, 0.004, headers=headers)
        got = basalt.read_trace_headers(path)
        full = basalt.read_segy(path)
    assert set(got) == set(_HEADER_DTYPES)
    for name, dtype in _HEADER_DTYPES.items():
        assert got[name].dtype == dtype, name
        assert got[name].shape == (4,), name
        assert np.array_equal(full["headers"][name], got[name]), name
    for name in ("trace_sequence", "field_record", "channel", "offset", "coord_units",
                 "delay_ms", "year", "day_of_year", "hour", "minute", "second"):
        assert got[name].tolist() == headers[name], name
    # A trace with a fraction in a coordinate gets -100; a trace whose coordinates are all whole
    # gets 1, so its values are exact.
    assert got["scalar_coord"].tolist() == [-100, -100, 1, -100]
    assert got["scalar_elev"].tolist() == [-100, -100, 1, 1]
    for name in ("source_x", "source_y", "group_x", "group_y", "receiver_elevation"):
        expected = np.array(headers[name], dtype=np.float64)
        assert np.max(np.abs(got[name] - expected)) <= 0.005 + 1e-9, name
    assert np.array_equal(full["samples"], samples)


def test_arcsecond_coordinates_for_station_near_lat_27_7_lon_85_3():
    # coord_units 2 means arcseconds: lon = group_x / 3600 and lat = group_y / 3600.
    # 27.7 and 85.3 are whole numbers of arcseconds. The second station has fractions.
    lat = np.array([27.7, 27.701234])
    lon = np.array([85.3, 85.309876])
    group_x = lon * 3600.0
    group_y = lat * 3600.0
    samples = np.zeros((2, 3), np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "arcsec.sgy")
        basalt.write_segy(path, samples, 0.004, headers={
            "coord_units": [2, 2],
            "source_x": group_x, "source_y": group_y,
            "group_x": group_x, "group_y": group_y,
        })
        got = basalt.read_trace_headers(path)
    assert got["coord_units"].tolist() == [2, 2]
    assert got["scalar_coord"].tolist() == [1, -100]
    assert got["group_x"][0] == 307080.0
    assert got["group_y"][0] == 99720.0
    assert np.max(np.abs(got["group_x"] - group_x)) <= 0.005 + 1e-9
    assert np.max(np.abs(got["group_y"] - group_y)) <= 0.005 + 1e-9
    # Converted back to degrees, the stored arcseconds give the station position.
    assert np.max(np.abs(got["group_x"] / 3600.0 - lon)) <= 0.005 / 3600.0 + 1e-12
    assert np.max(np.abs(got["group_y"] / 3600.0 - lat)) <= 0.005 / 3600.0 + 1e-12


def test_write_segy_time_fields_and_delay_round_trip():
    headers = {
        "year": [2026, 1999, -1, 32767],
        "day_of_year": [286, 366, 1, 0],
        "hour": [23, 0, 12, 6],
        "minute": [59, 1, 30, 45],
        "second": [59, 0, 1, 58],
        "delay_ms": [-1000, 0, 32767, -32768],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "time.sgy")
        basalt.write_segy(path, np.zeros((4, 2), np.float32), 0.004, headers=headers)
        got = basalt.read_trace_headers(path)
        for name, values in headers.items():
            assert got[name].dtype == np.int16, name
            assert got[name].tolist() == values, name
        # The time fields sit at bytes 157-166, and the delay at bytes 109-110.
        raw = _trace_header_bytes(path, 0, 2)
        assert struct.unpack_from(">h", raw, 156)[0] == 2026
        assert struct.unpack_from(">h", raw, 158)[0] == 286
        assert struct.unpack_from(">h", raw, 160)[0] == 23
        assert struct.unpack_from(">h", raw, 162)[0] == 59
        assert struct.unpack_from(">h", raw, 164)[0] == 59
        assert struct.unpack_from(">h", raw, 108)[0] == -1000


def test_write_segy_given_scalars_are_stored_and_applied():
    headers = {
        "scalar_coord": [10, -100, 0],
        "group_x": [1230.0, 123.45, 987654.0],
        "scalar_elev": [-10, 2, 0],
        "receiver_elevation": [-1.5, 500.0, 42.0],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scalars.sgy")
        basalt.write_segy(path, np.zeros((3, 2), np.float32), 0.004, headers=headers)
        got = basalt.read_trace_headers(path)
    assert got["scalar_coord"].tolist() == [10, -100, 0]
    assert got["scalar_elev"].tolist() == [-10, 2, 0]
    # With scalar 10 the stored value is 1230 / 10 = 123, and it reads back as 1230.
    assert got["group_x"][0] == 1230.0
    assert abs(got["group_x"][1] - 123.45) <= 0.005
    assert got["group_x"][2] == 987654.0
    assert got["receiver_elevation"].tolist() == [-1.5, 500.0, 42.0]

    # A scalar of 10 lets a value larger than the int32 range be stored.
    with tempfile.TemporaryDirectory() as tmp:
        big = os.path.join(tmp, "big.sgy")
        basalt.write_segy(big, np.zeros((1, 2), np.float32), 0.004,
                          headers={"scalar_coord": [10], "group_x": [1.0e10]})
        assert basalt.read_trace_headers(big)["group_x"][0] == 1.0e10


def test_write_segy_default_scalars_are_chosen_per_trace_and_absent_fields_stay_zero():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "defaults.sgy")
        basalt.write_segy(path, np.zeros((3, 2), np.float32), 0.004, headers={
            "group_x": [1.0, 2.5, -7.0],
            "receiver_elevation": [0.0, 0.0, 0.5],
        })
        got = basalt.read_trace_headers(path)
    assert got["scalar_coord"].tolist() == [1, -100, 1]
    assert got["scalar_elev"].tolist() == [1, 1, -100]
    assert got["group_x"].tolist() == [1.0, 2.5, -7.0]
    assert got["receiver_elevation"].tolist() == [0.0, 0.0, 0.5]
    # Fields that were not given keep their zero bytes.
    assert got["source_x"].tolist() == [0.0, 0.0, 0.0]
    assert got["source_y"].tolist() == [0.0, 0.0, 0.0]
    assert got["group_y"].tolist() == [0.0, 0.0, 0.0]


def test_write_segy_header_bytes_are_big_endian_at_contract_positions():
    headers = {
        "offset": [-1, 258],
        "group_y": [70000.0, -1.5],
        "hour": [5, 6],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bytes.sgy")
        basalt.write_segy(path, np.zeros((2, 3), np.float32), 0.004, headers=headers)
        first = _trace_header_bytes(path, 0, 3)
        second = _trace_header_bytes(path, 1, 3)
    # Bytes 37-40 hold offset, 85-88 group_y, 71-72 scalar_coord and 161-162 hour (1-based).
    assert struct.unpack_from(">i", first, 36)[0] == -1
    assert struct.unpack_from(">i", second, 36)[0] == 258
    assert struct.unpack_from(">i", first, 84)[0] == 70000
    assert struct.unpack_from(">i", second, 84)[0] == -150    # -1.5 at scalar -100
    assert struct.unpack_from(">h", first, 70)[0] == 1
    assert struct.unpack_from(">h", second, 70)[0] == -100
    assert struct.unpack_from(">h", first, 160)[0] == 5
    assert struct.unpack_from(">h", second, 160)[0] == 6
    # The trace sample count is still in bytes 115-116 and the sequence number in bytes 5-8.
    assert struct.unpack_from(">H", first, 114)[0] == 3
    assert struct.unpack_from(">i", second, 4)[0] == 2


def test_write_segy_headers_errors_raise_value_error_and_write_no_file():
    samples = np.zeros((3, 4), np.float32)
    bad = [
        {"source_x": [1.0, 2.0]},                       # wrong length
        {"source_x": [[1.0, 2.0, 3.0]]},                # not 1D
        {"source_x": 5.0},                              # scalar, not an array
        {"source_x": ["a", "b", "c"]},                  # not numbers
        {"sourcex": [1, 2, 3]},                         # unknown key
        {1: [1, 2, 3]},                                 # key that is not a str
        {"trace_sequence": [2 ** 31, 0, 0]},            # int32 overflow
        {"channel": [1.5, 2, 3]},                       # integer field, not whole
        {"year": [40000, 0, 0]},                        # int16 overflow
        {"coord_units": [-32769, 0, 0]},                # int16 underflow
        {"scalar_coord": [40000, 1, 1]},                # scalar beyond int16
        {"group_x": [3e9, 0, 0]},                       # int32 overflow at scalar 1
        {"scalar_coord": [-100, 1, 1], "group_x": [3e7, 0, 0]},  # overflow at scalar -100
        {"group_y": [float("nan"), 0, 0]},              # not finite
        {"receiver_elevation": [float("inf"), 0, 0]},   # not finite
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for i, headers in enumerate(bad):
            path = os.path.join(tmp, f"bad_{i}.sgy")
            _raises(ValueError, basalt.write_segy, path, samples, 0.004, headers=headers)
            assert not os.path.exists(path), headers
        path = os.path.join(tmp, "not_dict.sgy")
        _raises(ValueError, basalt.write_segy, path, samples, 0.004, headers=[1, 2, 3])
        assert not os.path.exists(path)
        # The same arrays are accepted once they fit.
        ok = os.path.join(tmp, "ok.sgy")
        basalt.write_segy(ok, samples, 0.004, headers={"scalar_coord": [100, 1, 1],
                                                       "group_x": [3e7, 0, 0]})
        assert basalt.read_trace_headers(ok)["group_x"][0] == 3e7


def test_write_segy_without_headers_or_with_empty_dict_writes_identical_files():
    rng = np.random.default_rng(43)
    samples = rng.standard_normal((2, 6)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        none_path = os.path.join(tmp, "none.sgy")
        empty_path = os.path.join(tmp, "empty.sgy")
        basalt.write_segy(none_path, samples, 0.002)
        basalt.write_segy(empty_path, samples, 0.002, headers={})
        with open(none_path, "rb") as a, open(empty_path, "rb") as b:
            assert a.read() == b.read()
        got = basalt.read_trace_headers(none_path)
    assert got["scalar_coord"].tolist() == [0, 0]
    assert got["scalar_elev"].tolist() == [0, 0]
    assert got["group_x"].tolist() == [0.0, 0.0]


def test_read_headers_dict_can_be_written_back_unchanged():
    headers = {
        "trace_sequence": [5, 6], "field_record": [9, 9], "channel": [1, 2],
        "offset": [-400, 400], "receiver_elevation": [1200.5, -3.0],
        "source_x": [-0.25, 10.0], "source_y": [3.5, -2.0],
        "group_x": [307080.0, 307155.56], "group_y": [99720.0, 99724.44],
        "coord_units": [2, 2], "delay_ms": [0, 12],
        "year": [2026, 2026], "day_of_year": [286, 287], "hour": [1, 2], "minute": [3, 4],
        "second": [5, 6],
    }
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.sgy")
        dst = os.path.join(tmp, "dst.sgy")
        basalt.write_segy(src, np.ones((2, 4), np.float32), 0.004, headers=headers)
        read = basalt.read_segy(src)
        basalt.write_segy(dst, read["samples"], read["dt"], read["format"], read["headers"])
        again = basalt.read_trace_headers(dst)
        redone = basalt.read_segy(dst)
    for name, dtype in _HEADER_DTYPES.items():
        assert again[name].dtype == dtype, name
        assert np.array_equal(again[name], read["headers"][name]), name
        assert np.array_equal(redone["headers"][name], read["headers"][name]), name
    assert np.array_equal(redone["samples"], read["samples"])


def test_write_segy_ibm_format_accepts_headers():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ibm.sgy")
        samples = np.array([[1.0, -2.5, 0.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        basalt.write_segy(path, samples, 0.004, format="ibm",
                          headers={"group_x": [307080.0, 12.25], "year": [2026, 2027]})
        got = basalt.read_trace_headers(path)
        full = basalt.read_segy(path)
    assert full["format"] == "ibm"
    assert got["year"].tolist() == [2026, 2027]
    assert got["scalar_coord"].tolist() == [1, -100]
    assert got["group_x"].tolist() == [307080.0, 12.25]
    assert np.allclose(full["samples"], samples)


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
