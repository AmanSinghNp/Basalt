#include "basalt/io/segy_writer.hpp"
#include "basalt/utils/ibm_float.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#define BASALT_GETPID _getpid
#else
#include <unistd.h>
#define BASALT_GETPID getpid
#endif

namespace basalt::io {

namespace {

constexpr size_t kFormatFieldOffset = SEGY_TEXT_HEADER_SIZE + 24; // data_sample_format

void store_u16(uint8_t* p, uint16_t v, bool le) {
    if (le) {
        p[0] = static_cast<uint8_t>(v & 0xFFu);
        p[1] = static_cast<uint8_t>(v >> 8);
    } else {
        p[0] = static_cast<uint8_t>(v >> 8);
        p[1] = static_cast<uint8_t>(v & 0xFFu);
    }
}

void store_u32(uint8_t* p, uint32_t v, bool le) {
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned shift = 8u * (le ? i : 3u - i);
        p[i] = static_cast<uint8_t>((v >> shift) & 0xFFu);
    }
}

std::string make_temp_path(const std::string& path) {
    static std::atomic<unsigned> counter{0};
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return path + ".basalt-tmp-" + std::to_string(static_cast<long>(BASALT_GETPID())) + "-" +
           std::to_string(static_cast<long long>(ticks)) + "-" +
           std::to_string(counter.fetch_add(1));
}

} // namespace

SegyWriter::SegyWriter(const std::string& path, const uint8_t* header_block, size_t header_bytes,
                       const SegyLayout& layout)
    : m_path(path), m_le(layout.little_endian), m_ns(layout.samples_per_trace),
      m_variable(layout.variable_length), m_expected_traces(layout.n_traces),
      m_trailer_bytes(layout.trailer_bytes) {
    if (header_block == nullptr || header_bytes != layout.data_offset ||
        header_bytes < SEGY_TEXT_HEADER_SIZE + SEGY_BINARY_HEADER_SIZE) {
        throw SegyWriteError("SEG-Y writer: header block must hold exactly layout.data_offset (" +
                             std::to_string(layout.data_offset) + ") bytes");
    }
    if (m_variable) {
        if (layout.trace_sample_counts.size() != layout.n_traces) {
            throw SegyWriteError("SEG-Y writer: variable-length layout has " +
                                 std::to_string(layout.trace_sample_counts.size()) +
                                 " sample counts for " + std::to_string(layout.n_traces) +
                                 " traces");
        }
        m_counts = layout.trace_sample_counts;
        for (uint32_t c : m_counts) {
            m_ns = std::max<size_t>(m_ns, c);
        }
    } else if (m_ns == 0) {
        throw SegyWriteError("SEG-Y writer: layout has zero samples per trace");
    }
    if (m_ns > SIZE_MAX / 4) {
        throw SegyWriteError("SEG-Y writer: samples per trace too large");
    }

    m_format = (layout.data_sample_format == 1) ? uint16_t{1} : uint16_t{5};
    m_buf.resize(m_ns * 4);

    m_tmp_path = make_temp_path(path);
    m_out.open(m_tmp_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!m_out) {
        throw SegyWriteError("cannot create '" + m_tmp_path + "' for writing");
    }

    try {
        if (m_format != layout.data_sample_format) {
            // Patch the format code in a private copy of the header block.
            std::vector<uint8_t> head(header_block, header_block + header_bytes);
            store_u16(head.data() + kFormatFieldOffset, m_format, m_le);
            write_bytes(head.data(), head.size());
        } else {
            write_bytes(header_block, header_bytes);
        }
    } catch (...) {
        discard();
        throw;
    }
}

SegyWriter::~SegyWriter() {
    if (!m_done) {
        discard();
    }
}

void SegyWriter::discard() noexcept {
    if (m_out.is_open()) {
        m_out.close();
    }
    std::error_code ec;
    std::filesystem::remove(m_tmp_path, ec);
    m_done = true;
}

void SegyWriter::write_bytes(const void* p, size_t n) {
    m_out.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    if (!m_out) {
        throw SegyWriteError("write to '" + m_tmp_path + "' failed (disk full or I/O error)");
    }
}

void SegyWriter::write_trace(const uint8_t* trace_header, const float* samples) {
    if (m_done) {
        throw SegyWriteError("SEG-Y writer: already committed or aborted");
    }
    if (m_written >= m_expected_traces) {
        throw SegyWriteError("SEG-Y writer: more traces than the layout declares (" +
                             std::to_string(m_expected_traces) + ")");
    }
    const size_t ns = m_variable ? m_counts[m_written] : m_ns;
    if (trace_header == nullptr || (samples == nullptr && ns != 0)) {
        throw SegyWriteError("SEG-Y writer: null trace header or samples");
    }
    try {
        write_bytes(trace_header, SEGY_TRACE_HEADER_SIZE);
        uint8_t* dst = m_buf.data();
        if (m_format == 1) {
            for (size_t k = 0; k < ns; ++k) {
                store_u32(dst + 4 * k, utils::ieee_to_ibm(samples[k]), m_le);
            }
        } else {
            for (size_t k = 0; k < ns; ++k) {
                uint32_t bits;
                std::memcpy(&bits, samples + k, sizeof bits);
                store_u32(dst + 4 * k, bits, m_le);
            }
        }
        write_bytes(dst, ns * 4);
    } catch (...) {
        discard();
        throw;
    }
    ++m_written;
}

void SegyWriter::commit(const uint8_t* trailer, size_t trailer_bytes) {
    if (m_done) {
        throw SegyWriteError("SEG-Y writer: already committed or aborted");
    }
    try {
        if (m_written != m_expected_traces) {
            throw SegyWriteError("SEG-Y writer: wrote " + std::to_string(m_written) +
                                 " traces but the layout declares " +
                                 std::to_string(m_expected_traces));
        }
        if (trailer_bytes != m_trailer_bytes || (trailer_bytes != 0 && trailer == nullptr)) {
            throw SegyWriteError("SEG-Y writer: trailer must be exactly " +
                                 std::to_string(m_trailer_bytes) + " bytes");
        }
        if (trailer_bytes != 0) {
            write_bytes(trailer, trailer_bytes);
        }
        m_out.flush();
        m_out.close();
        if (m_out.fail()) {
            throw SegyWriteError("closing '" + m_tmp_path + "' failed");
        }
        std::error_code ec;
        std::filesystem::rename(m_tmp_path, m_path, ec);
        if (ec) {
            throw SegyWriteError("cannot rename '" + m_tmp_path + "' to '" + m_path +
                                 "': " + ec.message());
        }
        m_done = true;
    } catch (...) {
        discard();
        throw;
    }
}

void write_segy_like(const std::string& path, const uint8_t* src, size_t src_size,
                     const SegyLayout& layout, const float* samples) {
    if (src == nullptr || src_size < layout.data_offset) {
        throw SegyWriteError("write_segy_like: source buffer shorter than the header block");
    }
    if (layout.n_traces != 0 && samples == nullptr) {
        throw SegyWriteError("write_segy_like: null samples");
    }
    // Everything the copy reads must lie inside the source buffer.
    const size_t avail = src_size - layout.data_offset;
    size_t trace_region = 0; // bytes from data_offset to the end of the last trace
    if (layout.variable_length) {
        if (layout.trace_offsets.size() != layout.n_traces ||
            layout.trace_sample_counts.size() != layout.n_traces) {
            throw SegyWriteError("write_segy_like: inconsistent variable-length layout");
        }
        for (size_t i = 0; i < layout.n_traces; ++i) {
            const uint64_t off = layout.trace_offsets[i];
            const uint64_t need =
                SEGY_TRACE_HEADER_SIZE +
                static_cast<uint64_t>(layout.trace_sample_counts[i]) * layout.bytes_per_sample;
            if (off < layout.data_offset || off > src_size || need > src_size - off) {
                throw SegyWriteError("write_segy_like: layout does not fit the source buffer");
            }
            trace_region = static_cast<size_t>(off + need - layout.data_offset);
        }
        if (avail < trace_region || avail - trace_region < layout.trailer_bytes) {
            throw SegyWriteError("write_segy_like: layout does not fit the source buffer");
        }
    } else {
        if (layout.trace_stride != 0 && layout.n_traces > SIZE_MAX / layout.trace_stride) {
            throw SegyWriteError("write_segy_like: layout size overflow");
        }
        trace_region = layout.n_traces * layout.trace_stride;
        if (avail < trace_region || avail - trace_region < layout.trailer_bytes) {
            throw SegyWriteError("write_segy_like: layout does not fit the source buffer");
        }
    }
    if (layout.n_traces != 0 && layout.samples_per_trace != 0 &&
        layout.n_traces > SIZE_MAX / layout.samples_per_trace) {
        throw SegyWriteError("write_segy_like: layout size overflow");
    }

    SegyWriter w(path, src, layout.data_offset, layout);
    for (size_t i = 0; i < layout.n_traces; ++i) {
        w.write_trace(src + layout.trace_offset(i), samples + i * layout.samples_per_trace);
    }
    w.commit(src + layout.data_offset + trace_region, layout.trailer_bytes);
}

} // namespace basalt::io
