#pragma once

// Internal interface shared by fk_pipeline.cpp (the single-gather pipeline) and fk_batch.cpp
// (fk_filter_gathers). Not part of the public API; see basalt/fk.hpp.

#include "basalt/fk.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::fk_internal {

// Padded geometry of one gather shape and the frequency and wavenumber steps (FK_PIPELINE.md,
// section 4, steps 2 and 5).
struct Geometry {
    size_t ntr_pad;  // padded trace count
    size_t ns_pad;   // padded sample count
    size_t nfreq;    // ns_pad / 2 + 1 (real-to-complex half spectrum)
    double df;       // frequency step, 1 / (ns_pad * dt)
    double dk;       // wavenumber step, 1 / (ntr_pad * dx)
};

// Validates `params` and the gather dimensions (the first two checks of fk_filter_gather). Throws
// std::invalid_argument with the same messages as fk_filter_gather.
void validate_arguments(const FKMuteParams& params, size_t ntraces, size_t nsamples);

// Validates `params` and the gather dimensions and computes the geometry. Throws
// std::invalid_argument with the same messages as fk_filter_gather.
Geometry check_geometry(const FKMuteParams& params, size_t ntraces, size_t nsamples);

// Mute weights and frequency column classes for one geometry. Read-only once built.
struct WeightTable {
    std::vector<float> weights;      // ntr_pad x nfreq, row i holds the bins of wavenumber index i
    std::vector<uint8_t> col_class;  // nfreq, one class per frequency column
};

// Builds the weight table for `geo` on `threads` threads (at least 1). The values do not depend on
// the thread count.
void build_weight_table(const FKMuteParams& params, const Geometry& geo, size_t threads,
                        WeightTable& table);

// Filters one gather on `threads` threads (at least 1), as fk_filter_gather does. When `shared`
// is null the weight table comes from the calling thread's cache. Otherwise `shared` must have
// been built by build_weight_table for the same params and gather shape, and it is only read.
// Same validation, aliasing and finiteness rules as fk_filter_gather, except that when
// `input_checked` is true the caller has already verified that every sample of `in` is finite
// (fk_filter_gathers scans the whole batch first), so the copy-in pass does not test them again.
// Passing true for input that was not scanned would let a non-finite sample through.
void filter_gather_impl(const float* in, float* out, size_t ntraces, size_t nsamples,
                        const FKMuteParams& params, size_t threads, const WeightTable* shared,
                        bool input_checked);

} // namespace basalt::fk_internal
