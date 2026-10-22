#include "basalt/io/file_mapping.hpp"
#include <cerrno>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace basalt::io {

namespace {

// "<what>: <path>: <os error text> (code N)"
std::string describe_error(const char* what, const std::string& path, int code) {
    return std::string(what) + ": " + path + ": " + std::system_category().message(code) +
           " (code " + std::to_string(code) + ")";
}

} // namespace

FileMapping::FileMapping(const std::string& path) {
    open_file(path);
}

FileMapping::~FileMapping() {
    cleanup();
}

FileMapping::FileMapping(FileMapping&& other) noexcept
    : m_data(other.m_data), m_size(other.m_size) {
    other.m_data = nullptr;
    other.m_size = 0;
}

FileMapping& FileMapping::operator=(FileMapping&& other) noexcept {
    if (this != &other) {
        cleanup();
        m_data = other.m_data;
        m_size = other.m_size;
        other.m_data = nullptr;
        other.m_size = 0;
    }
    return *this;
}

#ifdef _WIN32

namespace {

// Closes a HANDLE on scope exit unless released.
struct HandleGuard {
    HANDLE h;
    explicit HandleGuard(HANDLE handle) : h(handle) {}
    ~HandleGuard() {
        if (h != nullptr && h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
        }
    }
    HandleGuard(const HandleGuard&) = delete;
    HandleGuard& operator=(const HandleGuard&) = delete;
};

} // namespace

void FileMapping::open_file(const std::string& path) {
    // 1. Open File. Allow other processes to read and write while mapped.
    HandleGuard file(CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, // Hint for sequential access
        nullptr));

    if (file.h == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            describe_error("Failed to open file", path, static_cast<int>(GetLastError())));
    }

    // Get File Size
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file.h, &size)) {
        throw std::runtime_error(
            describe_error("Failed to get file size", path, static_cast<int>(GetLastError())));
    }
    if (size.QuadPart < 0 ||
        static_cast<unsigned long long>(size.QuadPart) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("File too large to map on this platform: " + path);
    }
    const size_t file_size = static_cast<size_t>(size.QuadPart);

    if (file_size == 0) {
        // Mapping a zero-length file fails on Windows
        m_data = nullptr;
        m_size = 0;
        return;
    }

    // 2. Create File Mapping Object
    HandleGuard mapping(CreateFileMappingA(file.h, nullptr, PAGE_READONLY, 0,
                                           0, // 0 here means "map whole file"
                                           nullptr));

    if (mapping.h == nullptr) {
        throw std::runtime_error(describe_error("Failed to create file mapping", path,
                                                static_cast<int>(GetLastError())));
    }

    // 3. Map View. The view keeps the file and mapping object alive, so both
    // handles are closed by the guards when this function returns.
    void* view = MapViewOfFile(mapping.h, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        throw std::runtime_error(
            describe_error("Failed to map view of file", path, static_cast<int>(GetLastError())));
    }
    m_data = view;
    m_size = file_size;
}

void FileMapping::cleanup() {
    if (m_data) {
        UnmapViewOfFile(m_data);
        m_data = nullptr;
    }
    m_size = 0;
}

void FileMapping::advise_sequential() {
    // On Windows, strictly speaking, we use FILE_FLAG_SEQUENTIAL_SCAN in CreateFile.
    // However, if we wanted to be more granular, we could use PrefetchVirtualMemory (Win8+).
    // For "Iron Foundation" compatibility (Windows 7+), CreateFile flag is sufficient.
}

#else

namespace {

// Closes a file descriptor on scope exit.
struct FdGuard {
    int fd;
    explicit FdGuard(int f) : fd(f) {}
    ~FdGuard() {
        if (fd != -1) {
            close(fd);
        }
    }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
};

} // namespace

void FileMapping::open_file(const std::string& path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    FdGuard file(open(path.c_str(), flags));
    if (file.fd == -1) {
        throw std::runtime_error(describe_error("Failed to open file", path, errno));
    }

    struct stat sb;
    if (fstat(file.fd, &sb) == -1) {
        throw std::runtime_error(describe_error("Failed to stat file", path, errno));
    }
    if (!S_ISREG(sb.st_mode)) {
        throw std::runtime_error("Failed to map file: " + path + ": not a regular file");
    }
    if (sb.st_size < 0 || static_cast<uint64_t>(sb.st_size) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("File too large to map on this platform: " + path);
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);

    if (file_size == 0) {
        m_data = nullptr;
        m_size = 0;
        return;
    }

    void* view = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (view == MAP_FAILED) {
        throw std::runtime_error(describe_error("Failed to mmap file", path, errno));
    }
    // The descriptor is closed by the guard: the mapping stays valid after close().
    m_data = view;
    m_size = file_size;
}

void FileMapping::cleanup() {
    if (m_data) {
        munmap(m_data, m_size);
        m_data = nullptr;
    }
    m_size = 0;
}

void FileMapping::advise_sequential() {
    if (m_data) {
        madvise(m_data, m_size, MADV_SEQUENTIAL);
    }
}

#endif

} // namespace basalt::io
