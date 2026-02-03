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

SegyTraceHeader parse_trace_header(const void* buffer) {
    SegyTraceHeader header;
    std::memcpy(&header, buffer, sizeof(SegyTraceHeader));

    // Swap key 32-bit fields
    header.trace_seq_line = swap_endian(header.trace_seq_line);
    header.trace_seq_file = swap_endian(header.trace_seq_file);
    header.field_record_number = swap_endian(header.field_record_number);
    header.trace_number_field = swap_endian(header.trace_number_field);
    header.energy_source_point = swap_endian(header.energy_source_point);
    header.cdp_ensemble_number = swap_endian(header.cdp_ensemble_number);
    header.trace_number_ensemble = swap_endian(header.trace_number_ensemble);
    header.source_receiver_offset = swap_endian(header.source_receiver_offset);
    header.receiver_elevation = swap_endian(header.receiver_elevation);
    header.source_elevation = swap_endian(header.source_elevation);
    header.source_depth = swap_endian(header.source_depth);
    header.datum_elev_receiver = swap_endian(header.datum_elev_receiver);
    header.datum_elev_source = swap_endian(header.datum_elev_source);
    header.water_depth_source = swap_endian(header.water_depth_source);
    header.water_depth_group = swap_endian(header.water_depth_group);
    header.source_x = swap_endian(header.source_x);
    header.source_y = swap_endian(header.source_y);
    header.group_x = swap_endian(header.group_x);
    header.group_y = swap_endian(header.group_y);

    // Swap key 16-bit fields
    header.trace_id_code = swap_endian(header.trace_id_code);
    header.num_vert_summed = swap_endian(header.num_vert_summed);
    header.num_horiz_stacked = swap_endian(header.num_horiz_stacked);
    header.data_use = swap_endian(header.data_use);
    header.scalar_elev = swap_endian(header.scalar_elev);
    header.scalar_coord = swap_endian(header.scalar_coord);
    header.coord_units = swap_endian(header.coord_units);
    header.weathering_velocity = swap_endian(header.weathering_velocity);
    header.subweathering_velocity = swap_endian(header.subweathering_velocity);
    header.uphole_time_source = swap_endian(header.uphole_time_source);
    header.uphole_time_group = swap_endian(header.uphole_time_group);
    header.source_static_corr = swap_endian(header.source_static_corr);
    header.group_static_corr = swap_endian(header.group_static_corr);
    header.total_static = swap_endian(header.total_static);
    header.lag_time_a = swap_endian(header.lag_time_a);
    header.lag_time_b = swap_endian(header.lag_time_b);
    header.delay_recording_time = swap_endian(header.delay_recording_time);
    header.mute_time_start = swap_endian(header.mute_time_start);
    header.mute_time_end = swap_endian(header.mute_time_end);
    header.samples_this_trace = swap_endian(header.samples_this_trace);
    header.sample_interval_us = swap_endian(header.sample_interval_us);

    return header;
}

} // namespace basalt::io
