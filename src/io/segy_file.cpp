#include "basalt/io/segy_file.hpp"
#include "basalt/io/segy_parser.hpp"
#include "basalt/segy_format.hpp"
#include "basalt/utils/ibm_float.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define BASALT_SEGY_AVX2 1
#else
#define BASALT_SEGY_AVX2 0
#endif

namespace basalt::io {

namespace {

// Layout constants. Offsets are 0-based within each header; the 1-based standard byte
// numbers are given in the comments (file offset = 3200 + (byte - 3201) for binary fields).
constexpr size_t kTextHeaderSize = SEGY_TEXT_HEADER_SIZE;       // 3200
constexpr size_t kBinaryHeaderSize = SEGY_BINARY_HEADER_SIZE;   // 400
constexpr size_t kFileHeaderSize = kTextHeaderSize + kBinaryHeaderSize;  // 3600
constexpr size_t kExtTextHeaderSize = 3200;
constexpr size_t kTraceHeaderSize = 240;

constexpr size_t kBinSampleIntervalOff = 16;   // bytes 3217-3218
constexpr size_t kBinSamplesOff = 20;          // bytes 3221-3222
constexpr size_t kBinFormatOff = 24;           // bytes 3225-3226
constexpr size_t kBinRevisionOff = 300;        // bytes 3501-3502
constexpr size_t kBinFixedLengthOff = 302;     // bytes 3503-3504
constexpr size_t kBinExtCountOff = 304;        // bytes 3505-3506

constexpr size_t kTrSeqLineOff = 0;            // bytes 1-4
constexpr size_t kTrSeqFileOff = 4;            // bytes 5-8
constexpr size_t kTrSamplesOff = 114;          // bytes 115-116
constexpr size_t kTrIntervalOff = 116;         // bytes 117-118

constexpr uint16_t kRevision1 = 0x0100;
constexpr uint64_t kMaxSamples = 0xFFFF;       // sample counts are 16-bit in the headers
constexpr size_t kMaxIntervalUs = 0xFFFF;      // sample interval is 16-bit microseconds

// Raw bytes moved per read or write call. Large enough to amortise the stream calls, small
// enough that the staging buffer stays a few megabytes whatever the file size.
constexpr size_t kChunkBytes = size_t{8} << 20;

uint16_t get_be16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint32_t>(p[0]) << 8) | static_cast<uint32_t>(p[1]));
}

uint32_t get_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
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

bool is_supported_format(uint16_t code) {
    return code == 1 || code == 2 || code == 3 || code == 5 || code == 8;
}

size_t bytes_per_sample(SampleFormat format) {
    switch (format) {
    case SampleFormat::IBM32:
    case SampleFormat::INT32:
    case SampleFormat::IEEE32:
        return 4;
    case SampleFormat::INT16:
        return 2;
    case SampleFormat::INT8:
        return 1;
    }
    return 4;
}

float decode_sample(const uint8_t* p, SampleFormat format) {
    switch (format) {
    case SampleFormat::IEEE32: {
        uint32_t bits = get_be32(p);
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    case SampleFormat::IBM32:
        return utils::ibm_to_ieee(get_be32(p));
    case SampleFormat::INT32:
        return static_cast<float>(static_cast<int32_t>(get_be32(p)));
    case SampleFormat::INT16:
        return static_cast<float>(static_cast<int16_t>(get_be16(p)));
    case SampleFormat::INT8:
        return static_cast<float>(static_cast<int8_t>(p[0]));
    }
    return 0.0f;
}

// Encodes a sample as big-endian IEEE32 or IBM32 (the only writable formats).
// ieee_to_ibm mis-scales IEEE subnormals (it assumes an implicit leading bit), so they are flushed
// to zero here (magnitude below 1.2e-38). Non-finite values are rejected by write_segy beforehand.
void encode_sample(uint8_t* p, float value, SampleFormat format) {
    uint32_t bits;
    if (format == SampleFormat::IBM32) {
        bits = (std::fpclassify(value) == FP_SUBNORMAL) ? 0u : utils::ieee_to_ibm(value);
    } else {
        std::memcpy(&bits, &value, sizeof(bits));
    }
    put_be32(p, bits);
}

// ---- Vector codec kernels --------------------------------------------------------------------
//
// The kernels below produce exactly the bits of utils::ibm_to_ieee / utils::ieee_to_ibm (and of a
// plain byte swap for IEEE32) for every input they accept; the scalar code is kept for the tails
// and for the INT formats. Derivation for IBM32 -> IEEE, with frac != 0 and p = index of the
// leading one of frac (0..23): normalised mantissa = frac << (23 - p), IEEE exponent =
// 4 * ibm_exp - 130 - (23 - p). Zero or underflowing exponents give +0.0, overflow gives
// +-infinity (the caller rejects that). For IEEE -> IBM32 with a normal float (biased exponent e in
// [1, 254]): IBM exponent = ceil((e + 130) / 4), mantissa = (implicit-1 significand) >> rem with
// rem = (-(e + 130)) mod 4, and e = 0 (zero or subnormal) encodes to 0.

#if BASALT_SEGY_AVX2

inline __m256i bswap32_lanes(__m256i v) {
    const __m256i mask = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                          3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    return _mm256_shuffle_epi8(v, mask);
}

#endif

// Decodes n big-endian IEEE32 words.
void decode_ieee_words(const uint8_t* src, float* dst, size_t n) {
    size_t i = 0;
#if BASALT_SEGY_AVX2
    for (; i + 8 <= n; i += 8) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 4 * i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), bswap32_lanes(v));
    }
#endif
    for (; i < n; ++i) {
        const uint32_t bits = get_be32(src + 4 * i);
        std::memcpy(dst + i, &bits, sizeof(float));
    }
}

// Decodes n big-endian IBM32 words. Returns the index of the first decoded value that is not
// finite (the file is then rejected), or n when every value is finite.
size_t decode_ibm_words(const uint8_t* src, float* dst, size_t n) {
    size_t i = 0;
    bool bad = false;
#if BASALT_SEGY_AVX2
    const __m256i frac_mask = _mm256_set1_epi32(0x00FFFFFF);
    const __m256i exp_mask = _mm256_set1_epi32(0x7F);
    const __m256i sign_bit = _mm256_set1_epi32(static_cast<int>(0x80000000u));
    const __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i c23 = _mm256_set1_epi32(23);
    const __m256i c127 = _mm256_set1_epi32(127);
    const __m256i c130 = _mm256_set1_epi32(130);
    const __m256i c254 = _mm256_set1_epi32(254);
    const __m256i c1 = _mm256_set1_epi32(1);
    const __m256i inf_bits = _mm256_set1_epi32(0x7F800000);
    const __m256i zero = _mm256_setzero_si256();
    __m256i any_over = zero;
    for (; i + 8 <= n; i += 8) {
        const __m256i w = bswap32_lanes(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 4 * i)));
        const __m256i frac = _mm256_and_si256(w, frac_mask);
        const __m256i ibm_exp = _mm256_and_si256(_mm256_srli_epi32(w, 24), exp_mask);
        const __m256i sign = _mm256_and_si256(w, sign_bit);
        // Leading-one position of frac. Exact because frac < 2^24 converts to float exactly.
        // For frac == 0 the lane is overwritten below.
        const __m256i fbits = _mm256_castps_si256(_mm256_cvtepi32_ps(frac));
        const __m256i p = _mm256_sub_epi32(_mm256_srli_epi32(fbits, 23), c127);
        const __m256i shift = _mm256_sub_epi32(c23, p);
        const __m256i mant = _mm256_and_si256(_mm256_sllv_epi32(frac, shift), mant_mask);
        const __m256i iexp = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_slli_epi32(ibm_exp, 2), c130), shift);

        const __m256i is_zero = _mm256_cmpeq_epi32(frac, zero);
        const __m256i is_over = _mm256_andnot_si256(is_zero, _mm256_cmpgt_epi32(iexp, c254));
        const __m256i is_under = _mm256_cmpgt_epi32(c1, iexp);  // iexp < 1
        __m256i res = _mm256_or_si256(_mm256_or_si256(sign, _mm256_slli_epi32(iexp, 23)), mant);
        res = _mm256_blendv_epi8(res, _mm256_or_si256(sign, inf_bits), is_over);
        res = _mm256_andnot_si256(_mm256_or_si256(is_zero, is_under), res);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), res);
        any_over = _mm256_or_si256(any_over, is_over);
    }
    bad = _mm256_testz_si256(any_over, any_over) == 0;
#endif
    for (; i < n; ++i) {
        dst[i] = utils::ibm_to_ieee(get_be32(src + 4 * i));
        bad = bad || !std::isfinite(dst[i]);
    }
    if (!bad) {
        return n;
    }
    for (size_t j = 0; j < n; ++j) {
        if (!std::isfinite(dst[j])) {
            return j;
        }
    }
    return n;
}

// Encodes n floats as big-endian IEEE32 words.
void encode_ieee_words(const float* in, uint8_t* dst, size_t n) {
    size_t i = 0;
#if BASALT_SEGY_AVX2
    for (; i + 8 <= n; i += 8) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 4 * i), bswap32_lanes(v));
    }
#endif
    for (; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, in + i, sizeof(bits));
        put_be32(dst + 4 * i, bits);
    }
}

// Encodes n finite floats as big-endian IBM32 words (see encode_sample for subnormals).
void encode_ibm_words(const float* in, uint8_t* dst, size_t n) {
    size_t i = 0;
#if BASALT_SEGY_AVX2
    const __m256i zero = _mm256_setzero_si256();
    const __m256i sign_bit = _mm256_set1_epi32(static_cast<int>(0x80000000u));
    const __m256i exp_mask = _mm256_set1_epi32(0xFF);
    const __m256i frac_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i implicit_one = _mm256_set1_epi32(0x00800000);
    const __m256i c3 = _mm256_set1_epi32(3);
    const __m256i c130 = _mm256_set1_epi32(130);
    for (; i + 8 <= n; i += 8) {
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        const __m256i e = _mm256_and_si256(_mm256_srli_epi32(b, 23), exp_mask);
        const __m256i sign = _mm256_and_si256(b, sign_bit);
        const __m256i mant = _mm256_or_si256(_mm256_and_si256(b, frac_mask), implicit_one);
        const __m256i temp = _mm256_add_epi32(e, c130);
        const __m256i ibm_exp = _mm256_srli_epi32(_mm256_add_epi32(temp, c3), 2);
        const __m256i rem = _mm256_and_si256(_mm256_sub_epi32(zero, temp), c3);
        const __m256i m = _mm256_srlv_epi32(mant, rem);
        __m256i res = _mm256_or_si256(_mm256_or_si256(sign, _mm256_slli_epi32(ibm_exp, 24)), m);
        res = _mm256_andnot_si256(_mm256_cmpeq_epi32(e, zero), res);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 4 * i), bswap32_lanes(res));
    }
#endif
    for (; i < n; ++i) {
        encode_sample(dst + 4 * i, in[i], SampleFormat::IBM32);
    }
}

// Decodes n samples of `format` from big-endian bytes. Returns the index of the first value that
// is not finite (IBM32 only), or n.
size_t decode_samples(const uint8_t* src, float* dst, size_t n, SampleFormat format) {
    switch (format) {
    case SampleFormat::IEEE32:
        decode_ieee_words(src, dst, n);
        return n;
    case SampleFormat::IBM32:
        return decode_ibm_words(src, dst, n);
    default: {
        const size_t width = bytes_per_sample(format);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = decode_sample(src + i * width, format);
        }
        return n;
    }
    }
}

// Encodes n samples as big-endian bytes in IEEE32 or IBM32.
void encode_samples(const float* in, uint8_t* dst, size_t n, SampleFormat format) {
    if (format == SampleFormat::IBM32) {
        encode_ibm_words(in, dst, n);
    } else {
        encode_ieee_words(in, dst, n);
    }
}

// ASCII text headers are recognised by counting printable bytes. EBCDIC space (0x40) is '@' in
// ASCII, so it is not counted as printable; otherwise a blank EBCDIC header would look like ASCII.
// NUL bytes are neutral: they are padding in some ASCII headers and say nothing about the encoding.
bool looks_like_ascii(const std::array<uint8_t, kTextHeaderSize>& text) {
    size_t considered = 0;
    size_t printable = 0;
    for (uint8_t b : text) {
        if (b == 0) {
            continue;
        }
        ++considered;
        const bool printable_ascii = (b >= 0x20 && b <= 0x7E && b != 0x40);
        if (printable_ascii || b == '\t' || b == '\n' || b == '\r') {
            ++printable;
        }
    }
    return printable * 2 > considered;
}

// ASCII to EBCDIC (code page 037) for the printable characters used in generated headers.
// Characters without a mapping become EBCDIC space.
uint8_t ascii_to_ebcdic(char c) {
    if (c >= 'A' && c <= 'I') return static_cast<uint8_t>(0xC1 + (c - 'A'));
    if (c >= 'J' && c <= 'R') return static_cast<uint8_t>(0xD1 + (c - 'J'));
    if (c >= 'S' && c <= 'Z') return static_cast<uint8_t>(0xE2 + (c - 'S'));
    if (c >= 'a' && c <= 'i') return static_cast<uint8_t>(0x81 + (c - 'a'));
    if (c >= 'j' && c <= 'r') return static_cast<uint8_t>(0x91 + (c - 'j'));
    if (c >= 's' && c <= 'z') return static_cast<uint8_t>(0xA2 + (c - 's'));
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(0xF0 + (c - '0'));
    switch (c) {
    case '.': return 0x4B;
    case '<': return 0x4C;
    case '(': return 0x4D;
    case '+': return 0x4E;
    case '|': return 0x4F;
    case '&': return 0x50;
    case '!': return 0x5A;
    case '$': return 0x5B;
    case '*': return 0x5C;
    case ')': return 0x5D;
    case ';': return 0x5E;
    case '-': return 0x60;
    case '/': return 0x61;
    case ',': return 0x6B;
    case '%': return 0x6C;
    case '_': return 0x6D;
    case '>': return 0x6E;
    case '?': return 0x6F;
    case ':': return 0x7A;
    case '#': return 0x7B;
    case '@': return 0x7C;
    case '\'': return 0x7D;
    case '=': return 0x7E;
    case '"': return 0x7F;
    default: return 0x40;
    }
}

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw std::runtime_error("SEG-Y file '" + path + "': " + what);
}

// True when `count` equals ntraces * nsamples. Compared by division so the product cannot wrap
// around for very large trace counts. nsamples must be nonzero (callers validate it first).
bool sample_count_matches(size_t ntraces, size_t nsamples, size_t count) {
    return nsamples != 0 && count % nsamples == 0 && count / nsamples == ntraces;
}

// Runs fn(begin, end) over a split of [0, n) into at most `threads` contiguous ranges. The calling
// thread runs the first range. If a worker thread cannot be started, its range runs on the calling
// thread instead, so the result does not depend on thread creation succeeding. An exception thrown
// by fn (for example std::bad_alloc while formatting an error) is captured on its thread and the
// first one is rethrown here after every range has finished, so it never escapes a thread.
template <typename Fn>
void parallel_ranges(size_t n, size_t threads, Fn&& fn) {
    const size_t nt = std::max<size_t>(1, std::min(threads, n));
    if (nt == 1) {
        fn(size_t{0}, n);
        return;
    }
    const size_t base = n / nt;
    const size_t extra = n % nt;
    const auto range_begin = [&](size_t i) { return i * base + std::min(i, extra); };
    std::mutex failure_mutex;
    std::exception_ptr failure;
    const auto run = [&fn, &failure_mutex, &failure](size_t b, size_t e) {
        try {
            fn(b, e);
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) {
                failure = std::current_exception();
            }
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(nt - 1);
    for (size_t i = 1; i < nt; ++i) {
        const size_t b = range_begin(i);
        const size_t e = range_begin(i + 1);
        try {
            workers.emplace_back([&run, b, e] { run(b, e); });
        } catch (...) {
            run(b, e);
        }
    }
    run(range_begin(0), range_begin(1));
    for (std::thread& w : workers) {
        w.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

size_t resolve_threads(size_t requested) {
    if (requested != 0) {
        return requested;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : hc;
}

// True when every value is finite. Checked over parallel ranges, so for a 512 MB IBM write the
// check costs a fraction of a serial scan; the answer does not depend on the thread count.
bool all_finite(const std::vector<float>& values, size_t threads) {
    std::atomic<bool> finite{true};
    parallel_ranges(values.size(), threads, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            if (!std::isfinite(values[i])) {
                finite.store(false, std::memory_order_relaxed);
                return;
            }
        }
    });
    return finite.load(std::memory_order_relaxed);
}

// Runs one task at a time on a helper thread, so file I/O can overlap with codec work. start() waits
// for the previous task before it begins the next. wait() joins the task and rethrows an exception it
// threw. If no thread can be started, the task runs on the calling thread instead, so the result is
// the same either way.
class BackgroundTask {
public:
    BackgroundTask() = default;
    BackgroundTask(const BackgroundTask&) = delete;
    BackgroundTask& operator=(const BackgroundTask&) = delete;
    ~BackgroundTask() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    template <typename Fn>
    void start(Fn fn) {
        wait();
        auto body = [this, fn]() mutable {
            try {
                fn();
            } catch (...) {
                failure_ = std::current_exception();
            }
        };
        try {
            worker_ = std::thread(body);
        } catch (const std::system_error&) {
            body();
        }
    }

    void wait() {
        if (worker_.joinable()) {
            worker_.join();
        }
        if (failure_) {
            const std::exception_ptr failure = failure_;
            failure_ = nullptr;
            std::rethrow_exception(failure);
        }
    }

private:
    std::thread worker_;
    std::exception_ptr failure_;
};

} // namespace

SegyData read_segy(const std::string& path, size_t threads) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open SEG-Y file '" + path + "'");
    }

    in.seekg(0, std::ios::end);
    const std::streamoff end_pos = in.tellg();
    in.seekg(0, std::ios::beg);
    if (end_pos < 0) {
        fail(path, "cannot determine file size");
    }
    const auto file_size = static_cast<uint64_t>(end_pos);
    if (file_size < kFileHeaderSize) {
        fail(path, "file is smaller than the 3600-byte textual and binary headers");
    }

    SegyData data;
    in.read(reinterpret_cast<char*>(data.raw_text_header.data()),
            static_cast<std::streamsize>(kTextHeaderSize));
    in.read(reinterpret_cast<char*>(data.raw_binary_header.data()),
            static_cast<std::streamsize>(kBinaryHeaderSize));
    if (!in) {
        fail(path, "unexpected end of file while reading headers");
    }

    const auto& bin = data.raw_binary_header;

    if (looks_like_ascii(data.raw_text_header)) {
        // text_header must be ASCII, so bytes outside 7-bit ASCII become spaces here.
        // raw_text_header keeps the original bytes for write-back.
        data.text_header.assign(kTextHeaderSize, ' ');
        for (size_t i = 0; i < kTextHeaderSize; ++i) {
            const uint8_t b = data.raw_text_header[i];
            if (b < 0x80) {
                data.text_header[i] = static_cast<char>(b);
            }
        }
    } else {
        data.text_header = parse_text_header(data.raw_text_header.data());
    }

    const uint16_t format_code = get_be16(&bin[kBinFormatOff]);
    if (!is_supported_format(format_code)) {
        fail(path, "unsupported sample format code " + std::to_string(format_code) +
                       " (supported: 1, 2, 3, 5, 8)");
    }
    data.format = static_cast<SampleFormat>(format_code);

    const auto ext_count = static_cast<int16_t>(get_be16(&bin[kBinExtCountOff]));
    if (ext_count == -1) {
        fail(path, "unsupported: variable number of extended textual headers (-1)");
    }
    if (ext_count < 0) {
        fail(path, "invalid extended textual header count " + std::to_string(ext_count));
    }
    const uint64_t data_start =
        kFileHeaderSize + static_cast<uint64_t>(ext_count) * kExtTextHeaderSize;
    if (file_size < data_start) {
        fail(path, "file is truncated: the " + std::to_string(ext_count) +
                       " extended textual header(s) extend past the end of the file");
    }
    in.seekg(static_cast<std::streamoff>(data_start), std::ios::beg);

    const size_t nsamples = get_be16(&bin[kBinSamplesOff]);
    if (nsamples == 0) {
        fail(path, "binary header declares zero samples per trace");
    }
    data.nsamples = nsamples;
    const uint16_t interval_us = get_be16(&bin[kBinSampleIntervalOff]);
    if (interval_us == 0) {
        fail(path, "binary header declares a zero sample interval");
    }
    data.dt = static_cast<double>(interval_us) * 1e-6;

    const size_t bps = bytes_per_sample(data.format);
    const uint64_t trace_bytes_u64 = kTraceHeaderSize + static_cast<uint64_t>(nsamples) * bps;
    const uint64_t payload = file_size - data_start;
    if (payload % trace_bytes_u64 != 0) {
        fail(path, "file size " + std::to_string(file_size) + " bytes does not match " +
                       std::to_string(data_start) + " bytes of headers followed by whole traces of " +
                       std::to_string(trace_bytes_u64) + " bytes (" + std::to_string(nsamples) +
                       " samples, format " + std::to_string(format_code) + ")");
    }

    const auto ntraces = static_cast<size_t>(payload / trace_bytes_u64);
    const auto trace_bytes = static_cast<size_t>(trace_bytes_u64);
    data.ntraces = ntraces;
    data.raw_trace_headers.resize(ntraces);
    data.samples.resize(ntraces * nsamples);

    // Traces are read in chunks of whole records and decoded in parallel. Each chunk is one read
    // call instead of two per trace. The next chunk is read on a helper thread while the current one
    // is decoded. Errors are recorded per trace and the first one in file order is reported, which
    // matches what a sequential read would report. If the file ends inside a chunk (it shrank after
    // the size check), that chunk's whole traces are still decoded and checked before the end of file
    // is reported at the first incomplete trace, as a sequential read would do.
    const size_t nthreads = resolve_threads(threads);
    const size_t per_chunk = std::max<size_t>(1, std::min(ntraces, kChunkBytes / trace_bytes));
    const size_t nchunks = (ntraces + per_chunk - 1) / per_chunk;
    const auto chunk_count = [&](size_t k) { return std::min(per_chunk, ntraces - k * per_chunk); };
    // Whole traces obtained by a read of `count` traces that returned `bytes_read` bytes.
    const auto whole_traces = [trace_bytes](size_t count, size_t bytes_read) {
        return std::min(count, bytes_read / trace_bytes);
    };
    std::vector<uint8_t> raw[2] = {std::vector<uint8_t>(per_chunk * trace_bytes),
                                   std::vector<uint8_t>(per_chunk * trace_bytes)};
    std::vector<std::string> errors(per_chunk);
    size_t next_got = 0;  // bytes the next-chunk read obtained (fewer than requested at end of file)
    BackgroundTask reader;  // declared last among the state its task uses, so it is joined first
    size_t avail = 0;       // whole traces in the current chunk
    if (nchunks > 0) {
        in.read(reinterpret_cast<char*>(raw[0].data()), static_cast<std::streamsize>(chunk_count(0) * trace_bytes));
        avail = whole_traces(chunk_count(0), static_cast<size_t>(in.gcount()));
    }
    for (size_t k = 0; k < nchunks; ++k) {
        const size_t first = k * per_chunk;
        const size_t count = chunk_count(k);
        const uint8_t* raw_block = raw[k % 2].data();
        if (k + 1 < nchunks) {
            uint8_t* next_block = raw[(k + 1) % 2].data();
            const size_t next_count = chunk_count(k + 1);
            reader.start([&in, &next_got, next_block, next_count, trace_bytes] {
                in.read(reinterpret_cast<char*>(next_block), static_cast<std::streamsize>(next_count * trace_bytes));
                next_got = static_cast<size_t>(in.gcount());
            });
        }
        std::fill(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(avail), std::string());

        parallel_ranges(avail, nthreads, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) {
                const size_t t = first + i;
                const uint8_t* record = raw_block + i * trace_bytes;
                auto& header = data.raw_trace_headers[t];
                std::memcpy(header.data(), record, kTraceHeaderSize);

                const uint16_t trace_samples = get_be16(&header[kTrSamplesOff]);
                if (trace_samples != 0 && trace_samples != nsamples) {
                    errors[i] = "trace " + std::to_string(t) + " header declares " +
                                std::to_string(trace_samples) +
                                " samples but the binary header declares " + std::to_string(nsamples);
                    continue;
                }

                float* out = data.samples.data() + t * nsamples;
                const size_t bad = decode_samples(record + kTraceHeaderSize, out, nsamples, data.format);
                // ibm_to_ieee returns infinity only for IBM words whose exponent is beyond the IEEE
                // range. Such a sample cannot be represented, so the file is rejected, not read as inf.
                if (bad < nsamples) {
                    errors[i] = "IBM sample " + std::to_string(bad) + " of trace " + std::to_string(t) +
                                " exceeds the IEEE single-precision range";
                }
            }
        });

        for (size_t i = 0; i < avail; ++i) {
            if (!errors[i].empty()) {
                fail(path, errors[i]);  // the reader is joined by its destructor
            }
        }
        if (avail < count) {
            fail(path, "unexpected end of file while reading trace " + std::to_string(first + avail));
        }
        reader.wait();
        if (k + 1 < nchunks) {
            avail = whole_traces(chunk_count(k + 1), next_got);
        }
    }

    return data;
}

void write_segy(const std::string& path, const SegyData& data, SampleFormat out_format, size_t threads) {
    if (out_format != SampleFormat::IBM32 && out_format != SampleFormat::IEEE32) {
        throw std::invalid_argument("write_segy: only IBM32 and IEEE32 output formats are supported");
    }
    if (data.nsamples == 0 || data.nsamples > kMaxSamples) {
        throw std::invalid_argument("write_segy: nsamples must be in [1, 65535]");
    }
    if (!sample_count_matches(data.ntraces, data.nsamples, data.samples.size())) {
        throw std::invalid_argument("write_segy: samples size does not equal ntraces * nsamples");
    }
    // Checked before any file is created, so a rejected write leaves no partial file behind.
    if (out_format == SampleFormat::IBM32 && !all_finite(data.samples, resolve_threads(threads))) {
        throw std::invalid_argument("write_segy: IBM32 cannot encode NaN or infinite samples");
    }
    if (data.raw_trace_headers.size() != data.ntraces) {
        throw std::invalid_argument("write_segy: one raw trace header is required per trace");
    }
    if (get_be16(&data.raw_binary_header[kBinSamplesOff]) != data.nsamples) {
        throw std::invalid_argument(
            "write_segy: binary header samples per trace does not match nsamples");
    }
    for (size_t t = 0; t < data.ntraces; ++t) {
        const uint16_t trace_samples = get_be16(&data.raw_trace_headers[t][kTrSamplesOff]);
        if (trace_samples != 0 && trace_samples != data.nsamples) {
            throw std::invalid_argument("write_segy: trace " + std::to_string(t) +
                                        " header samples do not match nsamples");
        }
    }

    std::array<uint8_t, kBinaryHeaderSize> bin = data.raw_binary_header;
    put_be16(&bin[kBinFormatOff], static_cast<uint16_t>(out_format));
    put_be16(&bin[kBinExtCountOff], 0);  // extended textual headers are not retained

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot create SEG-Y file '" + path + "'");
    }

    out.write(reinterpret_cast<const char*>(data.raw_text_header.data()),
              static_cast<std::streamsize>(kTextHeaderSize));
    out.write(reinterpret_cast<const char*>(bin.data()), static_cast<std::streamsize>(kBinaryHeaderSize));

    // Each chunk of whole trace records is encoded in parallel into one of two buffers, and each
    // finished chunk is written with one call on a helper thread. Encoding chunk k therefore overlaps
    // the file write of chunk k-1. A buffer is only reused after its write has completed.
    const size_t nthreads = resolve_threads(threads);
    const size_t bps = bytes_per_sample(out_format);
    const size_t trace_bytes = kTraceHeaderSize + data.nsamples * bps;
    const size_t per_chunk = std::max<size_t>(1, std::min(data.ntraces, kChunkBytes / trace_bytes));
    std::vector<uint8_t> buf[2] = {std::vector<uint8_t>(per_chunk * trace_bytes),
                                   std::vector<uint8_t>(per_chunk * trace_bytes)};
    BackgroundTask writer;  // declared after `out` and `buf`, so it is joined before either is destroyed
    for (size_t k = 0, first = 0; first < data.ntraces; ++k, first += per_chunk) {
        const size_t count = std::min(per_chunk, data.ntraces - first);
        uint8_t* block = buf[k % 2].data();
        parallel_ranges(count, nthreads, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) {
                const size_t t = first + i;
                uint8_t* record = block + i * trace_bytes;
                std::memcpy(record, data.raw_trace_headers[t].data(), kTraceHeaderSize);
                encode_samples(data.samples.data() + t * data.nsamples, record + kTraceHeaderSize,
                               data.nsamples, out_format);
            }
        });
        const auto bytes = static_cast<std::streamsize>(count * trace_bytes);
        writer.start([&out, block, bytes] { out.write(reinterpret_cast<const char*>(block), bytes); });
    }
    writer.wait();

    out.close();
    if (!out) {
        throw std::runtime_error("failed while writing SEG-Y file '" + path + "'");
    }
}

SegyData make_segy(const std::vector<float>& samples, size_t ntraces, size_t nsamples,
                   double dt, SampleFormat format) {
    if (format != SampleFormat::IBM32 && format != SampleFormat::IEEE32) {
        throw std::invalid_argument("make_segy: only IBM32 and IEEE32 formats are supported");
    }
    if (nsamples == 0 || nsamples > kMaxSamples) {
        throw std::invalid_argument("make_segy: nsamples must be in [1, 65535]");
    }
    if (!sample_count_matches(ntraces, nsamples, samples.size())) {
        throw std::invalid_argument("make_segy: samples size does not equal ntraces * nsamples");
    }
    if (!(dt > 0.0)) {
        throw std::invalid_argument("make_segy: dt must be positive");
    }
    // The SEG-Y sample interval is a 16-bit count of microseconds, so dt must be a whole number of
    // microseconds. A value that is not is rejected rather than silently rounded.
    const double scaled_us = dt * 1e6;
    const double interval_us = std::round(scaled_us);
    if (interval_us < 1.0 || interval_us > static_cast<double>(kMaxIntervalUs)) {
        throw std::invalid_argument("make_segy: dt must be between 1 us and 65535 us");
    }
    if (std::fabs(scaled_us - interval_us) > 1e-6) {
        throw std::invalid_argument("make_segy: dt must be a whole number of microseconds");
    }
    const auto dt_us = static_cast<uint16_t>(interval_us);

    SegyData data;
    data.format = format;
    data.ntraces = ntraces;
    data.nsamples = nsamples;
    data.dt = static_cast<double>(dt_us) * 1e-6;
    data.samples = samples;

    const std::string text = "C 1 CREATED BY BASALT";
    data.text_header.assign(kTextHeaderSize, ' ');
    data.text_header.replace(0, text.size(), text);
    for (size_t i = 0; i < kTextHeaderSize; ++i) {
        data.raw_text_header[i] = ascii_to_ebcdic(data.text_header[i]);
    }

    auto& bin = data.raw_binary_header;
    put_be16(&bin[kBinSampleIntervalOff], dt_us);
    put_be16(&bin[kBinSamplesOff], static_cast<uint16_t>(nsamples));
    put_be16(&bin[kBinFormatOff], static_cast<uint16_t>(format));
    put_be16(&bin[kBinRevisionOff], kRevision1);
    put_be16(&bin[kBinFixedLengthOff], 1);

    data.raw_trace_headers.resize(ntraces);
    for (size_t t = 0; t < ntraces; ++t) {
        auto& header = data.raw_trace_headers[t];
        header.fill(0);
        const auto seq = static_cast<uint32_t>(t + 1);
        put_be32(&header[kTrSeqLineOff], seq);
        put_be32(&header[kTrSeqFileOff], seq);
        put_be16(&header[kTrSamplesOff], static_cast<uint16_t>(nsamples));
        put_be16(&header[kTrIntervalOff], dt_us);
    }

    return data;
}

} // namespace basalt::io
