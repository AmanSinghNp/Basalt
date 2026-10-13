#include "basalt/io/trace_headers.hpp"
#include "basalt/io/segy_file.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace basalt::io {

namespace {

constexpr size_t kTraceHeaderSize = 240;

// Offsets are 0-based within a trace header; the 1-based SEG-Y bytes are in the comments.
constexpr size_t kTrSeqOff = 4;             // bytes 5-8
constexpr size_t kFieldRecordOff = 8;       // bytes 9-12
constexpr size_t kChannelOff = 12;          // bytes 13-16
constexpr size_t kOffsetOff = 36;           // bytes 37-40
constexpr size_t kRecvElevOff = 40;         // bytes 41-44
constexpr size_t kScalarElevOff = 68;       // bytes 69-70
constexpr size_t kScalarCoordOff = 70;      // bytes 71-72
constexpr size_t kSourceXOff = 72;          // bytes 73-76
constexpr size_t kSourceYOff = 76;          // bytes 77-80
constexpr size_t kGroupXOff = 80;           // bytes 81-84
constexpr size_t kGroupYOff = 84;           // bytes 85-88
constexpr size_t kCoordUnitsOff = 88;       // bytes 89-90
constexpr size_t kDelayOff = 108;           // bytes 109-110
constexpr size_t kYearOff = 156;            // bytes 157-158
constexpr size_t kDayOfYearOff = 158;       // bytes 159-160
constexpr size_t kHourOff = 160;            // bytes 161-162
constexpr size_t kMinuteOff = 162;          // bytes 163-164
constexpr size_t kSecondOff = 164;          // bytes 165-166

uint16_t be_u16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint32_t>(p[0]) << 8) | static_cast<uint32_t>(p[1]));
}

uint32_t be_u32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

int16_t be_i16(const uint8_t* p) {
    return static_cast<int16_t>(be_u16(p));
}

int32_t be_i32(const uint8_t* p) {
    return static_cast<int32_t>(be_u32(p));
}

// SEG-Y scalar rule: a positive scalar multiplies, a negative one divides by its absolute value,
// and zero means 1.
double apply_scalar(int32_t value, int16_t scalar) {
    const auto v = static_cast<double>(value);
    if (scalar > 0) {
        return v * static_cast<double>(scalar);
    }
    if (scalar < 0) {
        return v / -static_cast<double>(scalar);
    }
    return v;
}

void put_be16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v >> 8);
    p[1] = static_cast<uint8_t>(v & 0xFFu);
}

void put_be32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    p[2] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    p[3] = static_cast<uint8_t>(v & 0xFFu);
}

std::string format_number(double value) {
    std::ostringstream out;
    out.precision(12);
    out << value;
    return out.str();
}

// Value of an integer field, which must be a whole number in [lo, hi].
int64_t whole_value(double value, const char* name, int64_t lo, int64_t hi) {
    if (std::floor(value) != value) {
        throw std::invalid_argument("trace header field '" + std::string(name) +
                                    "' must be a whole number, got " + format_number(value));
    }
    if (value < static_cast<double>(lo) || value > static_cast<double>(hi)) {
        throw std::invalid_argument("trace header field '" + std::string(name) + "' value " +
                                    format_number(value) + " does not fit its field (" +
                                    std::to_string(lo) + ".." + std::to_string(hi) + ")");
    }
    return static_cast<int64_t>(value);
}

// Stores a real value as round(value / scale) in a signed 32-bit field. The scale is the SEG-Y
// scalar: a positive scalar divides by itself, a negative one multiplies by its absolute value,
// and zero means 1.
int32_t scaled_value(double value, int16_t scalar, const char* name) {
    double stored;
    if (scalar > 0) {
        stored = std::round(value / static_cast<double>(scalar));
    } else if (scalar < 0) {
        stored = std::round(value * -static_cast<double>(scalar));
    } else {
        stored = std::round(value);
    }
    constexpr double kMin = static_cast<double>(std::numeric_limits<int32_t>::min());
    constexpr double kMax = static_cast<double>(std::numeric_limits<int32_t>::max());
    if (!(stored >= kMin && stored <= kMax)) {
        throw std::invalid_argument("trace header field '" + std::string(name) + "' value " +
                                    format_number(value) +
                                    " overflows its 32-bit field with scalar " +
                                    std::to_string(scalar));
    }
    return static_cast<int32_t>(stored);
}

// The scalar of one trace. A given scalar is used as it is. Otherwise the default is -100 when a
// scaled value of this trace is not a whole number, and 1 when all of them are.
int16_t trace_scalar(const std::vector<double>& given,
                     const std::vector<const std::vector<double>*>& scaled, size_t t,
                     const char* name) {
    if (!given.empty()) {
        return static_cast<int16_t>(whole_value(given[t], name, std::numeric_limits<int16_t>::min(),
                                                std::numeric_limits<int16_t>::max()));
    }
    for (const std::vector<double>* values : scaled) {
        if (!values->empty() && std::floor((*values)[t]) != (*values)[t]) {
            return -100;
        }
    }
    return 1;
}

void store_int16(uint8_t* p, size_t offset, const std::vector<double>& values, size_t t,
                 const char* name) {
    if (values.empty()) {
        return;
    }
    const int64_t v = whole_value(values[t], name, std::numeric_limits<int16_t>::min(),
                                  std::numeric_limits<int16_t>::max());
    put_be16(p + offset, static_cast<uint16_t>(v));
}

void store_int32(uint8_t* p, size_t offset, const std::vector<double>& values, size_t t,
                 const char* name) {
    if (values.empty()) {
        return;
    }
    const int64_t v = whole_value(values[t], name, std::numeric_limits<int32_t>::min(),
                                  std::numeric_limits<int32_t>::max());
    put_be32(p + offset, static_cast<uint32_t>(v));
}

void store_scaled(uint8_t* p, size_t offset, const std::vector<double>& values, size_t t,
                  const char* name, int16_t scalar) {
    if (values.empty()) {
        return;
    }
    put_be32(p + offset, static_cast<uint32_t>(scaled_value(values[t], scalar, name)));
}

} // namespace

void TraceHeaders::resize(size_t n) {
    trace_sequence.resize(n);
    field_record.resize(n);
    channel.resize(n);
    offset.resize(n);
    scalar_elev.resize(n);
    scalar_coord.resize(n);
    coord_units.resize(n);
    source_x.resize(n);
    source_y.resize(n);
    group_x.resize(n);
    group_y.resize(n);
    receiver_elevation.resize(n);
    delay_ms.resize(n);
    year.resize(n);
    day_of_year.resize(n);
    hour.resize(n);
    minute.resize(n);
    second.resize(n);
}

TraceHeaders decode_trace_headers(const std::vector<std::array<uint8_t, kTraceHeaderSize>>& raw) {
    TraceHeaders out;
    out.resize(raw.size());
    for (size_t t = 0; t < raw.size(); ++t) {
        const uint8_t* p = raw[t].data();
        const int16_t scalar_elev = be_i16(p + kScalarElevOff);
        const int16_t scalar_coord = be_i16(p + kScalarCoordOff);

        out.trace_sequence[t] = be_i32(p + kTrSeqOff);
        out.field_record[t] = be_i32(p + kFieldRecordOff);
        out.channel[t] = be_i32(p + kChannelOff);
        out.offset[t] = be_i32(p + kOffsetOff);
        out.scalar_elev[t] = scalar_elev;
        out.scalar_coord[t] = scalar_coord;
        out.coord_units[t] = be_i16(p + kCoordUnitsOff);
        out.source_x[t] = apply_scalar(be_i32(p + kSourceXOff), scalar_coord);
        out.source_y[t] = apply_scalar(be_i32(p + kSourceYOff), scalar_coord);
        out.group_x[t] = apply_scalar(be_i32(p + kGroupXOff), scalar_coord);
        out.group_y[t] = apply_scalar(be_i32(p + kGroupYOff), scalar_coord);
        out.receiver_elevation[t] = apply_scalar(be_i32(p + kRecvElevOff), scalar_elev);
        out.delay_ms[t] = be_i16(p + kDelayOff);
        out.year[t] = be_i16(p + kYearOff);
        out.day_of_year[t] = be_i16(p + kDayOfYearOff);
        out.hour[t] = be_i16(p + kHourOff);
        out.minute[t] = be_i16(p + kMinuteOff);
        out.second[t] = be_i16(p + kSecondOff);
    }
    return out;
}

TraceHeaders read_trace_headers(const std::string& path) {
    return decode_trace_headers(read_segy_headers(path).raw_trace_headers);
}

void encode_trace_headers(std::vector<std::array<uint8_t, kTraceHeaderSize>>& raw,
                          const TraceHeaderValues& values) {
    const size_t ntraces = raw.size();
    const std::pair<const char*, const std::vector<double>*> fields[] = {
        {"trace_sequence", &values.trace_sequence},
        {"field_record", &values.field_record},
        {"channel", &values.channel},
        {"offset", &values.offset},
        {"scalar_elev", &values.scalar_elev},
        {"scalar_coord", &values.scalar_coord},
        {"coord_units", &values.coord_units},
        {"source_x", &values.source_x},
        {"source_y", &values.source_y},
        {"group_x", &values.group_x},
        {"group_y", &values.group_y},
        {"receiver_elevation", &values.receiver_elevation},
        {"delay_ms", &values.delay_ms},
        {"year", &values.year},
        {"day_of_year", &values.day_of_year},
        {"hour", &values.hour},
        {"minute", &values.minute},
        {"second", &values.second},
    };
    for (const auto& [name, vec] : fields) {
        if (vec->empty()) {
            continue;
        }
        if (vec->size() != ntraces) {
            throw std::invalid_argument("trace header field '" + std::string(name) + "' has " +
                                        std::to_string(vec->size()) + " values for " +
                                        std::to_string(ntraces) + " traces");
        }
        for (double v : *vec) {
            if (!std::isfinite(v)) {
                throw std::invalid_argument("trace header field '" + std::string(name) +
                                            "' must be finite, got " + format_number(v));
            }
        }
    }

    const std::vector<const std::vector<double>*> coords = {
        &values.source_x, &values.source_y, &values.group_x, &values.group_y};
    const std::vector<const std::vector<double>*> elevations = {&values.receiver_elevation};
    const bool coords_given = !values.source_x.empty() || !values.source_y.empty() ||
                              !values.group_x.empty() || !values.group_y.empty();
    const bool elevation_given = !values.receiver_elevation.empty();

    for (size_t t = 0; t < ntraces; ++t) {
        uint8_t* p = raw[t].data();
        store_int32(p, kTrSeqOff, values.trace_sequence, t, "trace_sequence");
        store_int32(p, kFieldRecordOff, values.field_record, t, "field_record");
        store_int32(p, kChannelOff, values.channel, t, "channel");
        store_int32(p, kOffsetOff, values.offset, t, "offset");

        if (elevation_given || !values.scalar_elev.empty()) {
            const int16_t scalar = trace_scalar(values.scalar_elev, elevations, t, "scalar_elev");
            put_be16(p + kScalarElevOff, static_cast<uint16_t>(scalar));
            store_scaled(p, kRecvElevOff, values.receiver_elevation, t, "receiver_elevation",
                         scalar);
        }
        if (coords_given || !values.scalar_coord.empty()) {
            const int16_t scalar = trace_scalar(values.scalar_coord, coords, t, "scalar_coord");
            put_be16(p + kScalarCoordOff, static_cast<uint16_t>(scalar));
            store_scaled(p, kSourceXOff, values.source_x, t, "source_x", scalar);
            store_scaled(p, kSourceYOff, values.source_y, t, "source_y", scalar);
            store_scaled(p, kGroupXOff, values.group_x, t, "group_x", scalar);
            store_scaled(p, kGroupYOff, values.group_y, t, "group_y", scalar);
        }

        store_int16(p, kCoordUnitsOff, values.coord_units, t, "coord_units");
        store_int16(p, kDelayOff, values.delay_ms, t, "delay_ms");
        store_int16(p, kYearOff, values.year, t, "year");
        store_int16(p, kDayOfYearOff, values.day_of_year, t, "day_of_year");
        store_int16(p, kHourOff, values.hour, t, "hour");
        store_int16(p, kMinuteOff, values.minute, t, "minute");
        store_int16(p, kSecondOff, values.second, t, "second");
    }
}

} // namespace basalt::io
