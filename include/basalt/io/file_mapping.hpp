#pragma once

#include <cstddef>
#include <string>

namespace basalt::io {

// Read-only memory mapping of a whole file.
//
// WARNING (external modification): the mapping is a live view of the file,
// not a snapshot. If the file is truncated or shrunk by another process while
// it is mapped, touching a page beyond the new end of file raises SIGBUS on
// POSIX (an access violation / EXCEPTION_IN_PAGE_ERROR on Windows) and
// terminates the process. MAP_PRIVATE does not protect against this, and
// writes by other processes to the existing range may or may not become
// visible. Only map files that are not modified for the lifetime of the
// mapping. Callers that cannot guarantee this should read the file into
// memory instead.
//
// A zero-length file is valid: size() == 0 and data() == nullptr.
//
// Errors are reported as std::runtime_error whose message includes the path
// and the operating system error (errno text on POSIX, GetLastError text on
// Windows).
class FileMapping {
public:
    explicit FileMapping(const std::string& path);
    ~FileMapping();

    // Prevent copying to avoid double-free of handle/map
    FileMapping(const FileMapping&) = delete;
    FileMapping& operator=(const FileMapping&) = delete;

    // Allow moving. The moved-from object is empty (data() == nullptr,
    // size() == 0). Self move-assignment is a no-op.
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
    // The file descriptor / file and mapping handles are closed as soon as the
    // view has been created (the mapping keeps the file alive), so nothing but
    // the view needs to be tracked.
};

} // namespace basalt::io
