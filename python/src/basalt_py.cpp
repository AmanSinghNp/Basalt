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
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "basalt/fk.hpp"
#include "basalt/io/segy_file.hpp"
#include "basalt/io/trace_headers.hpp"

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

// Filesystem path from str, bytes or any os.PathLike, via os.fspath. An embedded NUL is a
// ValueError, as in open(): the C file API would otherwise silently truncate the name.
std::string to_path(const py::handle path) {
    const py::object text = py::module_::import("os").attr("fspath")(path);
    std::string file = py::cast<std::string>(text);
    if (file.find('\0') != std::string::npos) {
        throw std::invalid_argument("path must not contain a NUL byte");
    }
    return file;
}

bool all_finite(const float* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return true;
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

    // A NaN or inf sample would spread through every bin of the transform and come back as an
    // all-NaN gather, so it is rejected up front.
    const float* in_ptr = gather.data();
    if (!all_finite(in_ptr, ntraces * nsamples)) {
        throw std::invalid_argument("fk_filter: data must contain only finite values (no NaN or inf)");
    }

    py::array_t<float> out(std::vector<py::ssize_t>{nrows, ncols});
    // `gather` holds a reference to the input buffer for the whole call, so the raw pointers
    // stay valid while the GIL is released. Nothing below touches Python objects.
    float* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        fk_filter_gather(in_ptr, out_ptr, ntraces, nsamples, params);
    }
    // Finite input can still overflow float32 inside the transform (for example at 1e36 scale).
    if (!all_finite(out_ptr, ntraces * nsamples)) {
        throw std::invalid_argument(
            "fk_filter: output is not finite; the data is too large for the float32 transform");
    }
    return out;
}

double mute_weight(double f_hz, double k_cpm, double v_min, double v_max, double taper,
                   const py::object& mode) {
    if (!(std::isfinite(f_hz) && std::isfinite(k_cpm))) {
        throw std::invalid_argument("fk_mute_weight: f_hz and k_cpm must be finite");
    }
    FKMuteParams params;
    params.v_min = v_min;
    params.v_max = v_max;
    params.taper = taper;
    params.mode = parse_mode(mode);
    return fk_mute_weight(f_hz, k_cpm, params);
}

// Copies a vector into a 1D array that owns its memory.
template <typename T>
py::array_t<T> to_array(const std::vector<T>& values) {
    py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return out;
}

// Per-trace header dict: one 1D array per field, one entry per trace.
py::dict trace_header_dict(const io::TraceHeaders& h) {
    py::dict out;
    out["trace_sequence"] = to_array(h.trace_sequence);
    out["field_record"] = to_array(h.field_record);
    out["channel"] = to_array(h.channel);
    out["offset"] = to_array(h.offset);
    out["scalar_elev"] = to_array(h.scalar_elev);
    out["scalar_coord"] = to_array(h.scalar_coord);
    out["coord_units"] = to_array(h.coord_units);
    out["source_x"] = to_array(h.source_x);
    out["source_y"] = to_array(h.source_y);
    out["group_x"] = to_array(h.group_x);
    out["group_y"] = to_array(h.group_y);
    out["receiver_elevation"] = to_array(h.receiver_elevation);
    out["delay_ms"] = to_array(h.delay_ms);
    out["year"] = to_array(h.year);
    out["day_of_year"] = to_array(h.day_of_year);
    out["hour"] = to_array(h.hour);
    out["minute"] = to_array(h.minute);
    out["second"] = to_array(h.second);
    return out;
}

py::dict read_file(const py::object& path) {
    const std::string file = to_path(path);
    io::SegyData data;
    io::TraceHeaders headers;
    {
        py::gil_scoped_release release;
        data = io::read_segy(file);
        headers = io::decode_trace_headers(data.raw_trace_headers);
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
    result["headers"] = trace_header_dict(headers);
    return result;
}

py::dict read_headers_file(const py::object& path) {
    const std::string file = to_path(path);
    io::TraceHeaders headers;
    {
        // Only the headers are read: the sample bytes are skipped, never decoded.
        py::gil_scoped_release release;
        headers = io::read_trace_headers(file);
    }
    return trace_header_dict(headers);
}

// Trace header keys that write_segy accepts, each with the TraceHeaderValues vector it fills.
using HeaderValuesMember = std::vector<double> io::TraceHeaderValues::*;
struct WritableHeader {
    const char* key;
    HeaderValuesMember member;
};

const WritableHeader kWritableHeaders[] = {
    {"trace_sequence", &io::TraceHeaderValues::trace_sequence},
    {"field_record", &io::TraceHeaderValues::field_record},
    {"channel", &io::TraceHeaderValues::channel},
    {"offset", &io::TraceHeaderValues::offset},
    {"scalar_elev", &io::TraceHeaderValues::scalar_elev},
    {"scalar_coord", &io::TraceHeaderValues::scalar_coord},
    {"coord_units", &io::TraceHeaderValues::coord_units},
    {"source_x", &io::TraceHeaderValues::source_x},
    {"source_y", &io::TraceHeaderValues::source_y},
    {"group_x", &io::TraceHeaderValues::group_x},
    {"group_y", &io::TraceHeaderValues::group_y},
    {"receiver_elevation", &io::TraceHeaderValues::receiver_elevation},
    {"delay_ms", &io::TraceHeaderValues::delay_ms},
    {"year", &io::TraceHeaderValues::year},
    {"day_of_year", &io::TraceHeaderValues::day_of_year},
    {"hour", &io::TraceHeaderValues::hour},
    {"minute", &io::TraceHeaderValues::minute},
    {"second", &io::TraceHeaderValues::second},
};

// Reads the optional `headers` argument of write_segy. None or an empty dict sets no header
// values. Keys must name a trace header field, and each value must be a 1D array of numbers. The
// core checks the lengths, whole-number requirements and field ranges.
io::TraceHeaderValues header_values_arg(const py::object& headers) {
    io::TraceHeaderValues values;
    if (headers.is_none()) {
        return values;
    }
    if (!py::isinstance<py::dict>(headers)) {
        throw std::invalid_argument("write_segy: headers must be a dict of arrays, or None");
    }
    using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
    for (auto item : py::reinterpret_borrow<py::dict>(headers)) {
        if (!py::isinstance<py::str>(item.first)) {
            throw std::invalid_argument("write_segy: headers keys must be str");
        }
        const std::string key = py::cast<std::string>(item.first);
        const WritableHeader* field = nullptr;
        for (const WritableHeader& candidate : kWritableHeaders) {
            if (key == candidate.key) {
                field = &candidate;
                break;
            }
        }
        if (field == nullptr) {
            throw std::invalid_argument("write_segy: unknown trace header key '" + key + "'");
        }
        const DoubleArray array = DoubleArray::ensure(item.second);
        if (!array) {
            throw std::invalid_argument("write_segy: headers['" + key +
                                        "'] must be an array of numbers");
        }
        if (array.ndim() != 1) {
            throw std::invalid_argument("write_segy: headers['" + key +
                                        "'] must be 1D, with one value per trace");
        }
        std::vector<double>& target = values.*(field->member);
        target.assign(array.data(), array.data() + array.size());
    }
    return values;
}

void write_file(const py::object& path, const py::object& samples, double dt,
                const py::object& format, const py::object& headers) {
    const std::string file = to_path(path);
    const FloatArray array = as_gather(samples, "write_segy", "samples");
    const io::SampleFormat out_format = parse_write_format(format);
    const io::TraceHeaderValues header_values = header_values_arg(headers);
    const size_t ntraces = static_cast<size_t>(array.shape(0));
    const size_t nsamples = static_cast<size_t>(array.shape(1));
    const std::vector<float> values(array.data(), array.data() + array.size());
    py::gil_scoped_release release;
    io::SegyData data = io::make_segy(values, ntraces, nsamples, dt, out_format);
    // Header values are checked before the file is created, so a rejected call leaves no file.
    io::encode_trace_headers(data.raw_trace_headers, header_values);
    io::write_segy(file, data, out_format);
}

} // namespace
} // namespace basalt

PYBIND11_MODULE(_basalt, m) {
    m.doc() = "basalt f-k mute filter and SEG-Y I/O (C++ core). Use the 'basalt' package.";

    m.def("fk_filter", &basalt::filter_gather,
          "Filter a 2D gather (ntraces, nsamples) in f-k space and return a new float32 array.\n\n"
          "data is converted to C-contiguous float32. Complex data keeps only the real part "
          "(numpy emits ComplexWarning). NaN or inf samples, and output that overflows float32 "
          "(for example data near 1e36), raise ValueError. Float parameters accept any real "
          "number (bool counts as 0 or 1); a non-numeric value raises TypeError.",
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
          "text_header, ntraces, nsamples and headers (the read_trace_headers dict).",
          py::arg("path"));

    m.def("read_trace_headers", &basalt::read_headers_file,
          "Read only the trace headers of a SEG-Y file (str or os.PathLike). Sample data is not "
          "read or decoded. Returns a dict of 1D NumPy arrays, one entry per trace, with the keys "
          "and dtypes listed in docs/design/FK_PIPELINE.md, section 7.",
          py::arg("path"));

    m.def("write_segy", &basalt::write_file,
          "Write a 2D array (ntraces, nsamples) as a SEG-Y file (str or os.PathLike) in 'ieee' or "
          "'ibm' format.\n\n"
          "headers is an optional dict of 1D arrays, one value per trace, for any subset of the "
          "trace header keys in docs/design/FK_PIPELINE.md, section 7. Coordinates and receiver "
          "elevation are real values, stored as round(value / scale). A scalar not given is chosen "
          "per trace: -100 when a value of that trace is not a whole number, otherwise 1. Raises "
          "ValueError for unknown keys, wrong lengths, non-whole integer values and values that "
          "overflow their field; no file is written then.",
          py::arg("path"), py::arg("samples"), py::arg("dt"), py::arg("format") = "ieee",
          py::arg("headers") = py::none());
}
