#pragma once

#include <cstddef>
#include <string>

namespace basalt::io {

class FileMapping {
public:
    explicit FileMapping(const std::string& path);
    ~FileMapping();

    // Prevent copying to avoid double-free of handle/map
    FileMapping(const FileMapping&) = delete;
    FileMapping& operator=(const FileMapping&) = delete;

    // Allow moving
    FileMapping(FileMapping&&) noexcept;
    FileMapping& operator=(FileMapping&&) noexcept;

    [[nodiscard]] const void* data() const { return m_data; }
    [[nodiscard]] size_t size() const { return m_size; }
    
    // Hint to OS that we will read sequentially
    void advise_sequential();

private:
    void open_file(const std::string& path);
    void cleanup();

    void* m_data = nullptr;
    size_t m_size = 0;

#ifdef _WIN32
    void* m_file_handle = nullptr; // HANDLE
    void* m_map_handle = nullptr;  // HANDLE
#else
    int m_fd = -1;
#endif
};

} // namespace basalt::io
