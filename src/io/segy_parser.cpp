#include "basalt/io/segy_parser.hpp"
#include "basalt/utils/ebcdic.hpp"
#include <cstring>

namespace basalt::io {

std::string parse_text_header(const void* buffer) {
    return utils::EbcdicConverter::to_ascii(buffer, SEGY_TEXT_HEADER_SIZE);
}

SegyBinaryHeader parse_binary_header(const void* buffer) {
    // Copy to struct to ensure alignment (safe for unaligned buffer)
    SegyBinaryHeader header;
    std::memcpy(&header, buffer, sizeof(SegyBinaryHeader));

    // Swap endianness (Big Endian -> Host Endian)
    header.job_id = swap_endian(header.job_id);
    header.line_number = swap_endian(header.line_number);
    header.reel_number = swap_endian(header.reel_number);
    header.traces_per_ensemble = swap_endian(header.traces_per_ensemble);
    header.aux_traces_per_ensemble = swap_endian(header.aux_traces_per_ensemble);
    header.sample_interval = swap_endian(header.sample_interval);
    header.sample_interval_orig = swap_endian(header.sample_interval_orig);
    header.samples_per_trace = swap_endian(header.samples_per_trace);
    header.samples_per_trace_orig = swap_endian(header.samples_per_trace_orig);
    header.data_sample_format = swap_endian(header.data_sample_format);
    header.ensemble_fold = swap_endian(header.ensemble_fold);
    header.trace_sorting_code = swap_endian(header.trace_sorting_code);
    header.vertical_sum_code = swap_endian(header.vertical_sum_code);
    header.sweep_frequency_start = swap_endian(header.sweep_frequency_start);
    header.sweep_frequency_end = swap_endian(header.sweep_frequency_end);
    header.sweep_length = swap_endian(header.sweep_length);
    header.sweep_type = swap_endian(header.sweep_type);
    header.trace_number_of_sweep_channel = swap_endian(header.trace_number_of_sweep_channel);
    header.sweep_trace_taper_length_start = swap_endian(header.sweep_trace_taper_length_start);
    header.sweep_trace_taper_length_end = swap_endian(header.sweep_trace_taper_length_end);
    header.taper_type = swap_endian(header.taper_type);
    header.correlated_data_traces = swap_endian(header.correlated_data_traces);
    header.binary_gain_recovered = swap_endian(header.binary_gain_recovered);
    header.amplitude_recovery_method = swap_endian(header.amplitude_recovery_method);
    header.measurement_system = swap_endian(header.measurement_system);
    header.impulse_signal_polarity = swap_endian(header.impulse_signal_polarity);
    header.vibratory_polarity_code = swap_endian(header.vibratory_polarity_code);

    return header;
}

} // namespace basalt::io
