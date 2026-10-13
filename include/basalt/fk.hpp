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
    size_t threads = 1;             // fk_filter_gather: threads for the FFT and the copy, weight
                                    // and table passes; 0 = hardware concurrency.
                                    // fk_filter_gathers:
                                    // threads that filter gathers in parallel. Each calling thread
                                    // keeps its work buffers and weight tables until it exits
                                    // (about 101 MB for a 2048 x 2048 gather, plus up to 38 MB when
                                    // the mixed columns form several runs). Batch worker threads
                                    // release theirs after 2 s without work (see
                                    // fk_filter_gathers). Peak memory of a batch call is in
                                    // fk_filter_gathers
};

/// Filters a trace-major gather (`data[t * nsamples + s]`) in f-k space.
/// Throws std::invalid_argument on invalid parameters, zero sizes, null pointers, a padded
/// grid whose element count would overflow size_t, or any NaN or infinite input sample
/// ("fk: input contains NaN or infinite samples"). The non-finite check runs before any
/// output is written, so `out` and, when aliased, `in` are unchanged when the call throws.
/// Finite input is not checked for float32 overflow of the spectrum. If a spectrum bin overflows
/// to infinity, the output may be NaN or infinite, and the skipped-column paths may return a
/// finite value where a full product would give NaN. Scale the data to avoid this.
/// `in` and `out` may alias (in-place filtering is allowed).
void fk_filter_gather(const float* in, float* out,
                      size_t ntraces, size_t nsamples,
                      const FKMuteParams& params);

/// Vector overload. Throws std::invalid_argument if `in.size() != ntraces * nsamples`.
std::vector<float> fk_filter_gather(const std::vector<float>& in,
                                    size_t ntraces, size_t nsamples,
                                    const FKMuteParams& params);

/// Filters `ngathers` consecutive gathers of identical shape. Gather g is the block
/// `in + g * ntraces * nsamples` (each block laid out as in fk_filter_gather), and it is written
/// to the same offset in `out`.
///
/// With ngathers >= 2, the result is bit-identical to calling fk_filter_gather with threads = 1
/// on each gather, for any params.threads. Gathers run in parallel on up to params.threads
/// threads (0 = hardware concurrency), and the calling thread is one of them. Each gather is
/// filtered on one thread. With two or more threads the weight table and column classes are built
/// once per call (on params.threads threads) and only read by the workers. With one thread every
/// gather runs on the calling thread, which reuses its cached table as fk_filter_gather does, so
/// no table is built per call. With ngathers == 1 the call is exactly fk_filter_gather(in, out,
/// ntraces, nsamples, params), so params.threads also parallelises that one gather.
///
/// Cost: before filtering, the whole input is scanned for non-finite samples (a memory-bound pass,
/// about 12 ms for 96 MB on one thread on the machine it was measured on). Each gather then skips
/// its own finiteness test, which costs about as much as the scan, so a call is not slower than
/// the fk_filter_gather loop it replaces. Measured on that machine, at threads = 1, batch and loop
/// times agree to within the run-to-run noise (about 3 %) at 1001 x 1500 x 16 and 240 x 1001 x 64.
/// A call with several threads also builds the weight table once, on those threads, so it is not
/// rebuilt per gather. A single-threaded call reuses the calling thread's cached table.
///
/// Memory: each thread that filters gathers holds its own work buffers while the call runs, as
/// fk_filter_gather does for its calling thread (about 27 MB per thread at 1001 x 1500, and about
/// 100 MB at 2048 x 2048 counting one weight table). With two or more threads the shared weight
/// table (ntr_pad x nfreq floats, about 7 MB at 1001 x 1500) is also held. Peak use therefore grows
/// with params.threads, and threads = 0 on a many-core host multiplies it. A worker that has been
/// idle for 2 s exits, and its buffers are freed with it.
///
/// ngathers == 0 is a no-op. The parameters, dimensions, pointers and padded geometry are still
/// validated, so the same invalid arguments throw whatever the gather count.
///
/// Throws std::invalid_argument on everything fk_filter_gather rejects, with the same message for
/// the same invalid arguments, and also when ngathers * ntraces * nsamples overflows size_t. Any
/// NaN or infinite sample in any gather throws ("fk: input contains NaN or infinite samples"). The
/// whole input is scanned, in parallel, before any gather is filtered, so no output is written and
/// an aliased `in` is unchanged. Calls may run concurrently from different threads.
///
/// `in` and `out` must either be the same pointer (in-place filtering) or not overlap.
void fk_filter_gathers(const float* in, float* out, size_t ngathers,
                       size_t ntraces, size_t nsamples, const FKMuteParams& params);

/// Mute weight of the f-k bin with frequency `f_hz` (Hz) and wavenumber `k_cpm`
/// (cycles per metre). Pure function of its arguments. Throws std::invalid_argument
/// on invalid parameters.
double fk_mute_weight(double f_hz, double k_cpm, const FKMuteParams& params);

/// Padded trace count used by the pipeline (good FFT size >= ceil(ntraces * pad_factor)).
size_t fk_padded_traces(size_t ntraces, const FKMuteParams& params);

/// Padded sample count used by the pipeline (good real FFT size >= ceil(nsamples * pad_factor)).
size_t fk_padded_samples(size_t nsamples, const FKMuteParams& params);

} // namespace basalt
