#pragma once

// End-to-end f-k mute pipeline: edge tapers, 2D real FFT, apparent-velocity
// band weighting and inverse transform. Contract: docs/design/FK_PIPELINE.md.

#include <cstddef>
#include <vector>

namespace basalt {

/// Reject: mute the velocity band and keep everything else.
/// Pass:   keep only the velocity band and mute everything else.
enum class FKMode { Reject, Pass };

/// Parameters for the f-k mute pipeline.
/// Identity configuration: mode Reject with an empty band (e.g. v_min = 1e29, v_max = 1e30),
/// taper 0 and no edge tapers returns the raw input. Pass with a huge band is not the identity:
/// the k = 0 row (v = +inf) lies outside every finite band and is removed by design.
struct FKMuteParams {
    double dt = 0.004;              // Sample interval (seconds)
    double dx = 25.0;               // Trace spacing (metres)
    double v_min = -1500.0;         // Lower band edge (m/s, may be negative)
    double v_max = 1500.0;          // Upper band edge (m/s, may be negative)
    double taper = 0.1;             // Fractional edge width, in [0, 1)
    FKMode mode = FKMode::Reject;
    double pad_factor = 1.5;        // >= 1.0; each axis padded to a good FFT size >= n * pad_factor
    size_t edge_taper_traces = 4;   // Cosine taper on the first/last N traces before the FFT
    size_t edge_taper_samples = 0;  // Cosine taper on the first/last N samples of each trace
    bool preserve_dc = true;        // Keep the (f = 0, k = 0) bin in both modes. If false, that
                                    // bin is evaluated at v = 0, since -f/k is undefined there
    size_t threads = 1;             // FFT threads; 0 = hardware concurrency
};

/// Filters a trace-major gather (`data[t * nsamples + s]`) in f-k space.
/// Throws std::invalid_argument on invalid parameters, zero sizes, null pointers or a padded
/// grid whose element count would overflow size_t.
/// `in` and `out` may alias (in-place filtering is allowed).
void fk_filter_gather(const float* in, float* out,
                      size_t ntraces, size_t nsamples,
                      const FKMuteParams& params);

/// Vector overload. Throws std::invalid_argument if `in.size() != ntraces * nsamples`.
std::vector<float> fk_filter_gather(const std::vector<float>& in,
                                    size_t ntraces, size_t nsamples,
                                    const FKMuteParams& params);

/// Mute weight of the f-k bin with frequency `f_hz` (Hz) and wavenumber `k_cpm`
/// (cycles per metre). Pure function of its arguments. Throws std::invalid_argument
/// on invalid parameters.
double fk_mute_weight(double f_hz, double k_cpm, const FKMuteParams& params);

/// Padded trace count used by the pipeline (good FFT size >= ceil(ntraces * pad_factor)).
size_t fk_padded_traces(size_t ntraces, const FKMuteParams& params);

/// Padded sample count used by the pipeline (good real FFT size >= ceil(nsamples * pad_factor)).
size_t fk_padded_samples(size_t nsamples, const FKMuteParams& params);

} // namespace basalt
