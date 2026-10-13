#include "basalt/io/trace_headers.hpp"
#include "basalt/io/segy_file.hpp"

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

} // namespace basalt::io
