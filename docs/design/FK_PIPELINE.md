# F-K Pipeline Design Contract

This document is the single source of truth for the end-to-end f-k filtering
pipeline (`basalt/fk.hpp`), the SEG-Y file reader/writer (`basalt/io/segy_file.hpp`),
the `basalt` CLI and the Python bindings. All implementations and tests must follow it.

## 1. Data layout

A *gather* is `ntraces x nsamples` float32, **trace-major** (row-major, one trace per row):
`data[t * nsamples + s]` is sample `s` of trace `t`. `dt` is the sample interval in
seconds, `dx` the trace spacing in metres.

## 2. Transform and sign convention

Forward transform is the standard DFT `X(k, f) = sum_x sum_t d(x, t) exp(-2*pi*i*(f*t + k*x))`
(same as `numpy.fft.fft2`). Frequencies follow `numpy.fft.fftfreq` / `rfftfreq`:
`f_j = j / (nsamples_pad * dt)` for `j = 0 .. nsamples_pad/2` (real-to-complex half spectrum),
`k_i = fftfreq(ntraces_pad, dx)[i]` (wrapped, negative for `i > ntraces_pad/2`).

**Apparent velocity** of an f-k sample is defined as

    v = -f / k        (m/s)

With this sign, a linear event whose arrival time *increases* with trace position,
`t(x) = t0 + x / V` with `V > 0`, maps to `v = +V`. An event whose time decreases with
`x` maps to `v = -V`. Tests must check this with synthetic dipping events.

Special cases:
- `k == 0, f != 0`: `v = +inf` (vertical / flat events). Treated as outside any finite band.
- `f == 0, k != 0`: `v = 0`.
- `f == 0, k == 0` (DC): weight is forced to 1 when `preserve_dc` is true (default).

Nyquist lines. The symmetry `v(-f,-k) = v(f,k)` fails on two lines, where a bin is its own
negative:
- k-Nyquist row (even `ntraces_pad`): the bin has `k = -K`, `K = 1/(2 dx)`, and
  `v(f, -K) = f / K` while `v(-f, -K)` is a different value. The weight is not Hermitian in `f`.
- f-Nyquist column (even `nsamples_pad`): the bin has `f = +F`, `F = 1/(2 dt)`. `+F` and `-F`
  are the same bin, so `v` flips sign with `k` and the weight is not Hermitian in `k`.

Both lines only matter in the aliasing regime (`|v| >= dx/dt`). The implementation applies
them bin by bin as the formula defines them (the rfft2/irfft2 convention, where the inverse takes
the real part). Other bins are Hermitian-symmetric.

## 3. Mute weight

Parameters: `v_min < v_max` (m/s, either may be negative), `taper` in `[0, 1)`,
`mode` in `{Reject, Pass}`.

Membership in the band `[v_min, v_max]` is smoothed with raised-cosine edges. For an edge
value `e`, the transition half-width is `h(e) = 0.5 * taper * |e|` (if `h == 0` the edge is hard).

    rise(v; e) = 0                         if v <= e - h
                 1                         if v >= e + h
                 0.5 - 0.5*cos(pi*(v - (e - h)) / (2h))   otherwise
    fall(v; e) = 1 - rise(v; e)
    inside(v)  = rise(v; v_min) * fall(v; v_max)          (inside(+-inf) = 0)

    weight = 1 - inside(v)   for Reject  (mute the band, keep the rest)
    weight = inside(v)       for Pass    (keep only the band)

Because `v(-f, -k) = v(f, k)` on all bins except the Nyquist lines of section 2, the weight is
Hermitian-symmetric there, and the real part of the inverse transform is taken (the output is
real). On the Nyquist lines the weight is applied as defined, bin by bin.

Hard edges (`taper = 0`) make the band half-open, `(v_min, v_max]`: `rise(v_min; v_min) = 0`
so `v = v_min` is excluded, and `fall(v_max; v_max) = 1` so `v = v_max` is included. The same
applies to `Pass` and `Reject`, which are complementary.

`basalt::fk_mute_weight(f_hz, k_cpm, params)` exposes this function for testing. It must be a
pure function of its arguments.

## 4. Pipeline (`basalt::fk_filter_gather`)

```cpp
namespace basalt {
enum class FKMode { Reject, Pass };

struct FKMuteParams {
    double dt = 0.004;              // s
    double dx = 25.0;               // m
    double v_min = -1500.0;         // m/s
    double v_max = 1500.0;          // m/s
    double taper = 0.1;             // fractional edge width, [0, 1)
    FKMode mode = FKMode::Reject;
    double pad_factor = 1.5;        // >= 1.0; pad each axis to good_size(ceil(n * pad_factor))
    size_t edge_taper_traces = 4;   // cosine taper applied to first/last N traces before FFT
    size_t edge_taper_samples = 0;  // cosine taper applied to first/last N samples of each trace
    bool preserve_dc = true;
    size_t threads = 1;             // FFT threads; 0 = hardware concurrency
};

// Throws std::invalid_argument on invalid params (dt <= 0, dx <= 0, v_min >= v_max,
// taper outside [0,1), pad_factor < 1, ntraces == 0, nsamples == 0, null pointers, or a
// padded grid whose element count ntr_pad * ns_pad would overflow size_t).
// `in` and `out` may alias (in-place is allowed).
void fk_filter_gather(const float* in, float* out,
                      size_t ntraces, size_t nsamples,
                      const FKMuteParams& params);

std::vector<float> fk_filter_gather(const std::vector<float>& in,
                                    size_t ntraces, size_t nsamples,
                                    const FKMuteParams& params);

double fk_mute_weight(double f_hz, double k_cpm, const FKMuteParams& params);

// Padded sizes the pipeline will use (exposed for tests/benchmarks).
size_t fk_padded_traces(size_t ntraces, const FKMuteParams& params);
size_t fk_padded_samples(size_t nsamples, const FKMuteParams& params);
}
```

Steps:
1. Validate params.
2. Padded sizes: `ntr_pad = pocketfft good_size_cmplx(ceil(ntraces * pad_factor))`,
   `ns_pad = good_size_real(ceil(nsamples * pad_factor))`. Any size works (no power-of-2 requirement).
3. Copy input into a zeroed `ntr_pad x ns_pad` buffer (top-left), applying the edge cosine
   tapers (Hann half-window: weight for the i-th trace from the edge, i = 0..N-1, is
   `0.5 - 0.5*cos(pi*(i + 1)/(N + 1))`). Tapers are not undone afterwards. The start and end
   tapers are independent. If a gather is shorter than 2N, both reach the same traces and their
   weights multiply (e.g. a single trace with N = 4 is scaled by `w(0)^2`, about 0.0091). Weights
   are computed in double, so any `N` up to `SIZE_MAX` is finite.
4. Real-to-complex 2D FFT with pocketfft (`r2c` over axes {0, 1}, time axis last), forward.
5. Multiply each complex bin by `fk_mute_weight(f_j, k_i, params)`. Weights must be computed
   once per call per (i, j) or cached; no per-bin trig in a hot loop beyond that.
6. Complex-to-real inverse with scale `1 / (ntr_pad * ns_pad)`.
7. Copy the top-left `ntraces x nsamples` block to `out`.

Required numerical behaviour:
- With `mode = Reject` and a band that contains no energy, output equals input to within
  1e-4 relative RMS for traces outside the edge-taper zone. This criterion is for inputs
  that are smooth along x, such as events with a Hann envelope across the gather. A
  hard-truncated linear event does not meet it: truncation at the gather ends leaks energy
  into every velocity band. Measured: 5.7e-3 for +-1000 m/s events (dx = 5 m, dt = 2 ms,
  band [-400, 400], default edge tapers). The leakage is inherent to the FFT, not an
  implementation error.
- An all-pass configuration (`mode = Pass, v_min = -1e30, v_max = 1e30`, taper 0, no edge
  tapers) returns `out(t, s) = in(t, s) - m(s) + M` to 1e-5 relative RMS, where
  `m(s) = sum_t in(t, s) / ntr_pad` and `M = sum(in) / (ntr_pad * ns_pad)` (sums over the
  valid block). The k = 0, f != 0 row has `v = +inf` (section 2), so Pass drops it. The
  trace mean `m(s)` is therefore removed, and the DC bin keeps `M`. On zero-mean input
  (`m(s) = 0` and `M = 0`) the output is the input.
- Non power-of-2 sizes such as 1001 x 1500 and 7 x 13 work.
- Synthetic test: two linear events, one at `+1000 m/s` and one at `-1000 m/s`. Rejecting
  `[500, 2000]` attenuates the +1000 event by >= 20 dB while the -1000 event loses < 1 dB
  (measured by RMS energy along each event's trajectory or by projecting onto each event),
  measured on the central 50% of traces. Use `dx = 5 m`, `dt = 0.002 s` and a 25 Hz Ricker
  wavelet so the events are not spatially aliased (aliasing starts at `f = V / (2 dx)` = 100 Hz).

## 5. SEG-Y file I/O (`basalt/io/segy_file.hpp`)

```cpp
namespace basalt::io {
enum class SampleFormat : uint16_t { IBM32 = 1, INT32 = 2, INT16 = 3, IEEE32 = 5, INT8 = 8 };

struct SegyData {
    std::string text_header;                        // 3200 chars, ASCII (converted from EBCDIC if needed)
    std::array<uint8_t, 3200> raw_text_header{};    // original bytes, written back verbatim
    std::array<uint8_t, 400> raw_binary_header{};   // original bytes
    std::vector<std::array<uint8_t, 240>> raw_trace_headers;
    SampleFormat format = SampleFormat::IEEE32;
    size_t ntraces = 0;
    size_t nsamples = 0;
    double dt = 0.0;                                // seconds, from binary header bytes 3217-3218 (us)
    std::vector<float> samples;                     // trace-major ntraces * nsamples
};

// Reads a big-endian SEG-Y rev 0/1 file with fixed trace length. Supports formats 1, 2, 3, 5, 8.
// Honors extended textual headers count (bytes 3505-3506) by skipping them.
// Throws std::runtime_error with a clear message on malformed input.
SegyData read_segy(const std::string& path);

// Writes a SEG-Y file. Headers are written back from the raw arrays, except that the
// binary header format code is updated to `out_format` and samples are encoded in it.
// Supports writing IBM32 and IEEE32.
void write_segy(const std::string& path, const SegyData& data,
                SampleFormat out_format = SampleFormat::IEEE32);

// Builds a minimal valid SegyData from samples (used by tests, the CLI and Python).
SegyData make_segy(const std::vector<float>& samples, size_t ntraces, size_t nsamples,
                   double dt, SampleFormat format = SampleFormat::IEEE32);
}
```

Use the existing `basalt/utils/ibm_float.hpp`, `basalt/utils/ebcdic.hpp` and
`basalt/segy_format.hpp` helpers. Do not use memory mapping in the reader (plain
`std::ifstream` is fine); correctness first.

## 6. CLI

```
basalt info   <in.segy>
basalt filter <in.segy> <out.segy> [--vmin V] [--vmax V] [--taper T] [--mode reject|pass]
                                     [--dx M] [--dt S] [--pad P] [--edge-taper N]
                                     [--threads N] [--gather-size N] [--format ibm|ieee]
basalt --version | --help
```

- `dt` defaults to the file's binary header; `--dt` overrides. `dx` defaults to 25 m.
- `--gather-size N` filters the file in consecutive blocks of N traces (default: whole file
  as one gather).
- Output format defaults to the input format when it is IBM32 or IEEE32, else IEEE32.
- Exit code 0 on success, 2 on usage errors, 1 on runtime errors (message on stderr).

## 7. Python

Package `basalt` built with pybind11 (CMake option `BASALT_BUILD_PYTHON`, default OFF):

```python
import basalt
out = basalt.fk_filter(data, dt, dx, v_min=-1500, v_max=1500, taper=0.1,
                       mode="reject", pad_factor=1.5, edge_taper_traces=4,
                       edge_taper_samples=0, preserve_dc=True, threads=1)
w = basalt.fk_mute_weight(f_hz, k_cpm, v_min=..., v_max=..., taper=..., mode=...)
d = basalt.read_segy(path)   # dict: samples (ntraces x nsamples float32 ndarray), dt, format, text_header
basalt.write_segy(path, samples, dt, format="ieee")
```

`data` is a 2D float32 array of shape `(ntraces, nsamples)` (C-contiguous; other dtypes
are converted). Returns a new float32 array.
