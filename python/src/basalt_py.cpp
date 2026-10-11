// pybind11 bindings for the basalt f-k mute pipeline and SEG-Y file I/O.
// Contract: docs/design/FK_PIPELINE.md section 7. The module is `basalt._basalt`; the
// public package re-exports these functions from python/basalt/__init__.py.
//
// Argument policy: string, path, count and array arguments are taken as py::object and checked
// here, so wrong types and values raise ValueError (paths follow os.fspath and raise TypeError
// for non-path objects). Floating-point arguments keep pybind11's conversion.

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

// Reads a str argument. Any other object type is a ValueError, like an unknown string.
std::string str_arg(const py::handle value, const char* name) {
    if (!py::isinstance<py::str>(value)) {
        throw std::invalid_argument(std::string(name) + " must be a str");
    }
    return py::cast<std::string>(value);
}

FKMode parse_mode(const py::handle mode) {
    const std::string text = str_arg(mode, "mode");
    const std::string m = to_lower_ascii(text);
    if (m == "reject") {
        return FKMode::Reject;
    }
    if (m == "pass") {
        return FKMode::Pass;
    }
    throw std::invalid_argument("mode must be 'reject' or 'pass', got '" + text + "'");
}

// Converts an integer argument (threads, edge tapers) to a size in 0 .. 2**64-1. Any object with
// __index__ is accepted (Python int, numpy integers); bool, float and str are rejected, and so
// are negative values and values of 2**64 or more. All of these are ValueError.
size_t to_count(const py::handle value, const char* name) {
    if (py::isinstance<py::bool_>(value) || !PyIndex_Check(value.ptr())) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    const py::object index = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!index) {
        throw py::error_already_set();
    }
    // PyLong_AsUnsignedLongLong raises OverflowError for negative values and values >= 2**64.
    // The error is cleared and reported as ValueError; ULLONG_MAX itself is a valid count.
    const unsigned long long count = PyLong_AsUnsignedLongLong(index.ptr());
    if (count == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        PyErr_Clear();
        throw std::invalid_argument(std::string(name) + " must be an integer in 0..2**64-1");
    }
    return static_cast<size_t>(count);
}

// Filesystem path from str, bytes or any os.PathLike, via os.fspath.
std::string to_path(const py::handle path) {
    const py::object text = py::module_::import("os").attr("fspath")(path);
    return py::cast<std::string>(text);
}

// 2D float32 gather from any array-like. Ragged lists, strings and other values that do not
// convert to a numeric array are ValueError, as are arrays that are not 2D.
FloatArray as_gather(const py::handle value, const char* func, const char* name) {
    FloatArray array = FloatArray::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(func) + ": " + name +
                                    " must be a 2D array-like of numbers, shape (ntraces, nsamples)");
    }
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string(func) + ": " + name +
                                    " must be a 2D array of shape (ntraces, nsamples), got " +
                                    std::to_string(array.ndim()) + " dimensions");
    }
    return array;
}

io::SampleFormat parse_write_format(const py::handle format) {
    const std::string text = str_arg(format, "format");
    const std::string f = to_lower_ascii(text);
    if (f == "ieee") {
        return io::SampleFormat::IEEE32;
    }
    if (f == "ibm") {
        return io::SampleFormat::IBM32;
    }
    throw std::invalid_argument("format must be 'ieee' or 'ibm', got '" + text + "'");
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

py::array_t<float> filter_gather(const py::object& data, double dt, double dx, double v_min,
                                 double v_max, double taper, const py::object& mode,
                                 double pad_factor, const py::object& edge_taper_traces,
                                 const py::object& edge_taper_samples, bool preserve_dc,
                                 const py::object& threads) {
    const FloatArray gather = as_gather(data, "fk_filter", "data");
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

    const py::ssize_t nrows = gather.shape(0);
    const py::ssize_t ncols = gather.shape(1);
    const size_t ntraces = static_cast<size_t>(nrows);
    const size_t nsamples = static_cast<size_t>(ncols);

    py::array_t<float> out(std::vector<py::ssize_t>{nrows, ncols});
    // `gather` holds a reference to the input buffer for the whole call, so the raw pointers
    // stay valid while the GIL is released. Nothing below touches Python objects.
    const float* in_ptr = gather.data();
    float* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        fk_filter_gather(in_ptr, out_ptr, ntraces, nsamples, params);
    }
    return out;
}

double mute_weight(double f_hz, double k_cpm, double v_min, double v_max, double taper,
                   const py::object& mode) {
    FKMuteParams params;
    params.v_min = v_min;
    params.v_max = v_max;
    params.taper = taper;
    params.mode = parse_mode(mode);
    return fk_mute_weight(f_hz, k_cpm, params);
}

py::dict read_file(const py::object& path) {
    const std::string file = to_path(path);
    io::SegyData data;
    {
        py::gil_scoped_release release;
        data = io::read_segy(file);
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

void write_file(const py::object& path, const py::object& samples, double dt,
                const py::object& format) {
    const std::string file = to_path(path);
    const FloatArray array = as_gather(samples, "write_segy", "samples");
    const io::SampleFormat out_format = parse_write_format(format);
    const size_t ntraces = static_cast<size_t>(array.shape(0));
    const size_t nsamples = static_cast<size_t>(array.shape(1));
    const std::vector<float> values(array.data(), array.data() + array.size());
    py::gil_scoped_release release;
    const io::SegyData data = io::make_segy(values, ntraces, nsamples, dt, out_format);
    io::write_segy(file, data, out_format);
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
          py::arg("edge_taper_traces") = 4, py::arg("edge_taper_samples") = 0,
          py::arg("preserve_dc") = true, py::arg("threads") = 1);

    m.def("fk_mute_weight", &basalt::mute_weight,
          "Mute weight of the f-k bin with frequency f_hz (Hz) and wavenumber k_cpm (cycles/m).",
          py::arg("f_hz"), py::arg("k_cpm"),
          py::arg("v_min") = -1500.0, py::arg("v_max") = 1500.0, py::arg("taper") = 0.1,
          py::arg("mode") = "reject");

    m.def("read_segy", &basalt::read_file,
          "Read a SEG-Y file (str or os.PathLike). Returns a dict with samples, dt, format, "
          "text_header, ntraces, nsamples.",
          py::arg("path"));

    m.def("write_segy", &basalt::write_file,
          "Write a 2D array (ntraces, nsamples) as a SEG-Y file (str or os.PathLike) in 'ieee' or "
          "'ibm' format.",
          py::arg("path"), py::arg("samples"), py::arg("dt"), py::arg("format") = "ieee");
}
