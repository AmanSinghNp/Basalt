// pybind11 bindings for the basalt f-k mute pipeline and SEG-Y file I/O.
// Contract: docs/design/FK_PIPELINE.md section 7. The module is `basalt._basalt`; the
// public package re-exports these functions from python/basalt/__init__.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cctype>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "basalt/fk.hpp"
#include "basalt/io/segy_file.hpp"

namespace py = pybind11;

namespace basalt {
namespace {

// Float32 C-contiguous view of any array-like. Other dtypes and memory orders are converted.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::string to_lower_ascii(const std::string& text) {
    std::string out(text);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

FKMode parse_mode(const std::string& mode) {
    const std::string m = to_lower_ascii(mode);
    if (m == "reject") {
        return FKMode::Reject;
    }
    if (m == "pass") {
        return FKMode::Pass;
    }
    throw std::invalid_argument("mode must be 'reject' or 'pass', got '" + mode + "'");
}

// Converts a non-negative Python integer. Negative values are reported as ValueError here,
// instead of the TypeError pybind11 raises for unsigned C++ parameters.
size_t to_count(long long value, const char* name) {
    if (value < 0) {
        throw std::invalid_argument(std::string(name) + " must be >= 0");
    }
    return static_cast<size_t>(value);
}

io::SampleFormat parse_write_format(const std::string& format) {
    const std::string f = to_lower_ascii(format);
    if (f == "ieee") {
        return io::SampleFormat::IEEE32;
    }
    if (f == "ibm") {
        return io::SampleFormat::IBM32;
    }
    throw std::invalid_argument("format must be 'ieee' or 'ibm', got '" + format + "'");
}

const char* format_name(io::SampleFormat format) {
    switch (format) {
        case io::SampleFormat::IBM32:
            return "ibm";
        case io::SampleFormat::INT32:
            return "int32";
        case io::SampleFormat::INT16:
            return "int16";
        case io::SampleFormat::IEEE32:
            return "ieee";
        case io::SampleFormat::INT8:
            return "int8";
    }
    throw std::runtime_error("unknown SEG-Y sample format");
}

py::array_t<float> filter_gather(FloatArray data, double dt, double dx, double v_min,
                                 double v_max, double taper, const std::string& mode,
                                 double pad_factor, long long edge_taper_traces,
                                 long long edge_taper_samples, bool preserve_dc,
                                 long long threads) {
    if (data.ndim() != 2) {
        throw py::value_error("fk_filter: data must be a 2D array of shape (ntraces, nsamples), "
                              "got " + std::to_string(data.ndim()) + " dimensions");
    }
    FKMuteParams params;
    params.dt = dt;
    params.dx = dx;
    params.v_min = v_min;
    params.v_max = v_max;
    params.taper = taper;
    params.mode = parse_mode(mode);
    params.pad_factor = pad_factor;
    params.edge_taper_traces = to_count(edge_taper_traces, "edge_taper_traces");
    params.edge_taper_samples = to_count(edge_taper_samples, "edge_taper_samples");
    params.preserve_dc = preserve_dc;
    params.threads = to_count(threads, "threads");

    const py::ssize_t nrows = data.shape(0);
    const py::ssize_t ncols = data.shape(1);
    const size_t ntraces = static_cast<size_t>(nrows);
    const size_t nsamples = static_cast<size_t>(ncols);

    py::array_t<float> out(std::vector<py::ssize_t>{nrows, ncols});
    // `data` holds a reference to the input buffer for the whole call, so the raw pointers
    // stay valid while the GIL is released. Nothing below touches Python objects.
    const float* in_ptr = data.data();
    float* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        fk_filter_gather(in_ptr, out_ptr, ntraces, nsamples, params);
    }
    return out;
}

double mute_weight(double f_hz, double k_cpm, double v_min, double v_max, double taper,
                   const std::string& mode) {
    FKMuteParams params;
    params.v_min = v_min;
    params.v_max = v_max;
    params.taper = taper;
    params.mode = parse_mode(mode);
    return fk_mute_weight(f_hz, k_cpm, params);
}

py::dict read_file(const std::string& path) {
    io::SegyData data;
    {
        py::gil_scoped_release release;
        data = io::read_segy(path);
    }
    // Copy into an array that owns its memory, so the result does not depend on the C++ data.
    py::array_t<float> samples(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(data.ntraces), static_cast<py::ssize_t>(data.nsamples)});
    if (!data.samples.empty()) {
        std::memcpy(samples.mutable_data(), data.samples.data(),
                    data.samples.size() * sizeof(float));
    }
    py::dict result;
    result["samples"] = samples;
    result["dt"] = data.dt;
    result["format"] = format_name(data.format);
    result["text_header"] = data.text_header;
    result["ntraces"] = data.ntraces;
    result["nsamples"] = data.nsamples;
    return result;
}

void write_file(const std::string& path, FloatArray samples, double dt,
                const std::string& format) {
    if (samples.ndim() != 2) {
        throw py::value_error(
            "write_segy: samples must be a 2D array of shape (ntraces, nsamples), "
            "got " + std::to_string(samples.ndim()) + " dimensions");
    }
    const io::SampleFormat out_format = parse_write_format(format);
    const size_t ntraces = static_cast<size_t>(samples.shape(0));
    const size_t nsamples = static_cast<size_t>(samples.shape(1));
    const std::vector<float> values(samples.data(), samples.data() + samples.size());
    py::gil_scoped_release release;
    const io::SegyData data = io::make_segy(values, ntraces, nsamples, dt, out_format);
    io::write_segy(path, data, out_format);
}

} // namespace
} // namespace basalt

PYBIND11_MODULE(_basalt, m) {
    m.doc() = "basalt f-k mute filter and SEG-Y I/O (C++ core). Use the 'basalt' package.";

    m.def("fk_filter", &basalt::filter_gather,
          "Filter a 2D gather (ntraces, nsamples) in f-k space and return a new float32 array.",
          py::arg("data"), py::arg("dt"), py::arg("dx"),
          py::arg("v_min") = -1500.0, py::arg("v_max") = 1500.0, py::arg("taper") = 0.1,
          py::arg("mode") = "reject", py::arg("pad_factor") = 1.5,
          py::arg("edge_taper_traces") = 4LL, py::arg("edge_taper_samples") = 0LL,
          py::arg("preserve_dc") = true, py::arg("threads") = 1LL);

    m.def("fk_mute_weight", &basalt::mute_weight,
          "Mute weight of the f-k bin with frequency f_hz (Hz) and wavenumber k_cpm (cycles/m).",
          py::arg("f_hz"), py::arg("k_cpm"),
          py::arg("v_min") = -1500.0, py::arg("v_max") = 1500.0, py::arg("taper") = 0.1,
          py::arg("mode") = "reject");

    m.def("read_segy", &basalt::read_file,
          "Read a SEG-Y file. Returns a dict with samples, dt, format, text_header, ntraces, "
          "nsamples.",
          py::arg("path"));

    m.def("write_segy", &basalt::write_file,
          "Write a 2D float array (ntraces, nsamples) as a SEG-Y file in 'ieee' or 'ibm' format.",
          py::arg("path"), py::arg("samples"), py::arg("dt"), py::arg("format") = "ieee");
}
