#include "basalt/io/file_mapping.hpp"
#include <stdexcept>
#include <utility>

#ifdef _WIN32
    #define NOMINMAX
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace basalt::io {

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

#ifdef _WIN32
    m_file_handle = other.m_file_handle;
    m_map_handle = other.m_map_handle;
    other.m_file_handle = nullptr; // INVALID_HANDLE_VALUE equivalent would be better but nullptr works if checked
    other.m_map_handle = nullptr;
#else
    m_fd = other.m_fd;
    other.m_fd = -1;
#endif
}

FileMapping& FileMapping::operator=(FileMapping&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        m_data = other.m_data;
        m_size = other.m_size;
        
        other.m_data = nullptr;
        other.m_size = 0;

#ifdef _WIN32
        m_file_handle = other.m_file_handle;
        m_map_handle = other.m_map_handle;
        other.m_file_handle = nullptr;
        other.m_map_handle = nullptr;
#else
        m_fd = other.m_fd;
        other.m_fd = -1;
#endif
    }
    return *this;
}

#ifdef _WIN32

void FileMapping::open_file(const std::string& path) {
    // 1. Open File
    m_file_handle = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, // Hint for sequential access
        nullptr
    );

    if (m_file_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // Get File Size
    LARGE_INTEGER size;
    if (!GetFileSizeEx(m_file_handle, &size)) {
        CloseHandle(m_file_handle);
        throw std::runtime_error("Failed to get file size: " + path);
    }
    m_size = static_cast<size_t>(size.QuadPart);

    if (m_size == 0) {
        // Mapping a zero-length file fails on Windows
        m_data = nullptr;
        return; 
    }

    // 2. Create File Mapping Object
    m_map_handle = CreateFileMappingA(
        m_file_handle,
        nullptr,
        PAGE_READONLY,
        0, 
        0, // 0 here means "map whole file"
        nullptr
    );

    if (m_map_handle == nullptr) {
        CloseHandle(m_file_handle);
        throw std::runtime_error("Failed to create file mapping: " + path);
    }

    // 3. Map View
    m_data = MapViewOfFile(
        m_map_handle,
        FILE_MAP_READ,
        0, 0, 0
    );

    if (m_data == nullptr) {
        CloseHandle(m_map_handle);
        CloseHandle(m_file_handle);
        throw std::runtime_error("Failed to map view of file");
    }
}

void FileMapping::cleanup() {
    if (m_data) {
        UnmapViewOfFile(m_data);
        m_data = nullptr;
    }
    if (m_map_handle) {
        CloseHandle(m_map_handle);
        m_map_handle = nullptr;
    }
    if (m_file_handle && m_file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(m_file_handle);
        m_file_handle = nullptr; // or INVALID_HANDLE_VALUE
    }
}

void FileMapping::advise_sequential() {
    // On Windows, strictly speaking, we use FILE_FLAG_SEQUENTIAL_SCAN in CreateFile.
    // However, if we wanted to be more granular, we could use PrefetchVirtualMemory (Win8+).
    // For "Iron Foundation" compatibility (Windows 7+), CreateFile flag is sufficient.
}

#else

void FileMapping::open_file(const std::string& path) {
    m_fd = open(path.c_str(), O_RDONLY);
    if (m_fd == -1) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    struct stat sb;
    if (fstat(m_fd, &sb) == -1) {
        close(m_fd);
        throw std::runtime_error("Failed to stat file");
    }
    m_size = static_cast<size_t>(sb.st_size);

    if (m_size == 0) {
        m_data = nullptr;
        return;
    }

    m_data = mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
    if (m_data == MAP_FAILED) {
        close(m_fd);
        throw std::runtime_error("Failed to mmap file");
    }
}

void FileMapping::cleanup() {
    if (m_data && m_data != MAP_FAILED) {
        munmap(m_data, m_size);
        m_data = nullptr;
    }
    if (m_fd != -1) {
        close(m_fd);
        m_fd = -1;
    }
}

void FileMapping::advise_sequential() {
    if (m_data) {
        madvise(m_data, m_size, MADV_SEQUENTIAL);
    }
}

#endif

} // namespace basalt::io
