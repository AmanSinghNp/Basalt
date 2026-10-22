#include "basalt/io/file_mapping.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace basalt::io;
namespace fs = std::filesystem;

namespace {

// Creates a unique directory under the system temp dir and removes it on destruction.
class TempDir {
public:
    TempDir() {
        static std::atomic<unsigned> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
        const unsigned long pid = 0;
#else
        const unsigned long pid = static_cast<unsigned long>(::getpid());
#endif
        m_path = fs::temp_directory_path() /
                 ("basalt_filemap_" + std::to_string(pid) + "_" + std::to_string(stamp) + "_" +
                  std::to_string(counter++));
        fs::create_directories(m_path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(m_path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] std::string file(const std::string& name) const {
        return (m_path / name).string();
    }

private:
    fs::path m_path;
};

void write_file(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
}

} // namespace

class FileMappingTest : public ::testing::Test {
protected:
    void SetUp() override {
        filename = dir.file("test_map_file.bin");
        const std::string msg = "Hello from Memory Map!";
        file_size = msg.size();
        write_file(filename, msg);
    }

    TempDir dir;
    std::string filename;
    size_t file_size = 0;
};

TEST_F(FileMappingTest, OpenAndRead) {
    FileMapping mapping(filename);

    EXPECT_EQ(mapping.size(), file_size);
    EXPECT_NE(mapping.data(), nullptr);

    const char* data = static_cast<const char*>(mapping.data());
    std::string content(data, file_size);
    EXPECT_EQ(content, "Hello from Memory Map!");
}

TEST_F(FileMappingTest, MoveSemantics) {
    FileMapping mapping1(filename);
    EXPECT_NE(mapping1.data(), nullptr);

    // Move construction
    FileMapping mapping2(std::move(mapping1));
    EXPECT_EQ(mapping1.data(), nullptr); // original should be empty
    EXPECT_EQ(mapping1.size(), 0u);
    EXPECT_NE(mapping2.data(), nullptr); // new should be valid
    EXPECT_EQ(mapping2.size(), file_size);

    const char* data = static_cast<const char*>(mapping2.data());
    std::string content(data, file_size);
    EXPECT_EQ(content, "Hello from Memory Map!");
}

TEST_F(FileMappingTest, MoveAssignment) {
    const std::string other_name = dir.file("other.bin");
    write_file(other_name, "second file contents");

    FileMapping a(filename);
    FileMapping b(other_name);
    const void* b_data = b.data();

    a = std::move(b); // releases a's old mapping, takes b's
    EXPECT_EQ(a.data(), b_data);
    EXPECT_EQ(a.size(), std::string("second file contents").size());
    EXPECT_EQ(std::string(static_cast<const char*>(a.data()), a.size()), "second file contents");
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0u);

    // Assigning into a moved-from object works too.
    b = std::move(a);
    EXPECT_NE(b.data(), nullptr);
    EXPECT_EQ(a.data(), nullptr);
}

TEST_F(FileMappingTest, SelfMoveAssignmentIsNoOp) {
    FileMapping m(filename);
    const void* before = m.data();
    FileMapping& alias = m;
    m = std::move(alias);
    EXPECT_EQ(m.data(), before);
    EXPECT_EQ(m.size(), file_size);
    EXPECT_EQ(std::string(static_cast<const char*>(m.data()), m.size()), "Hello from Memory Map!");
}

TEST_F(FileMappingTest, ZeroLengthFile) {
    const std::string empty = dir.file("empty.bin");
    write_file(empty, "");
    ASSERT_TRUE(fs::exists(empty));

    FileMapping m(empty);
    EXPECT_EQ(m.size(), 0u);
    EXPECT_EQ(m.data(), nullptr);
    EXPECT_NO_THROW(m.advise_sequential());

    // Moving an empty mapping is fine as well.
    FileMapping moved(std::move(m));
    EXPECT_EQ(moved.size(), 0u);
    EXPECT_EQ(moved.data(), nullptr);

    FileMapping real(filename);
    real = std::move(moved); // replaces a live mapping with an empty one
    EXPECT_EQ(real.size(), 0u);
    EXPECT_EQ(real.data(), nullptr);
}

TEST_F(FileMappingTest, FileNotFound) {
    EXPECT_THROW({
        FileMapping mapping(dir.file("non_existent_file.bin"));
    }, std::exception);
}

TEST_F(FileMappingTest, MissingFileMessageHasPathAndOsError) {
    const std::string missing = dir.file("definitely_missing.bin");
    try {
        FileMapping mapping(missing);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find(missing), std::string::npos) << what;
#ifdef _WIN32
        EXPECT_NE(what.find("code 2"), std::string::npos) << what; // ERROR_FILE_NOT_FOUND
#else
        EXPECT_NE(what.find(std::system_category().message(ENOENT)), std::string::npos) << what;
        EXPECT_NE(what.find("code " + std::to_string(ENOENT)), std::string::npos) << what;
#endif
    }
}

#ifndef _WIN32
TEST_F(FileMappingTest, DirectoryIsRejected) {
    // open(2) on a directory succeeds with O_RDONLY, so the type must be checked.
    EXPECT_THROW({ FileMapping m(dir.file("")); }, std::runtime_error);
}

TEST_F(FileMappingTest, DoesNotLeakFileDescriptors) {
    // Count open descriptors via /proc when available; a leak of one fd per
    // mapping would be visible after many iterations.
    const fs::path proc_fd("/proc/self/fd");
    if (!fs::exists(proc_fd)) {
        GTEST_SKIP() << "/proc/self/fd not available";
    }
    auto count_fds = [&]() {
        size_t n = 0;
        for (const auto& e : fs::directory_iterator(proc_fd)) {
            (void)e;
            ++n;
        }
        return n;
    };
    const size_t before = count_fds();
    {
        std::vector<FileMapping> maps;
        for (int i = 0; i < 20; ++i) {
            maps.emplace_back(filename);
        }
        // The descriptor is closed right after mmap: mappings stay valid, no fds held.
        EXPECT_LE(count_fds(), before + 1);
        for (const auto& m : maps) {
            EXPECT_EQ(std::string(static_cast<const char*>(m.data()), m.size()),
                      "Hello from Memory Map!");
        }
    }
    EXPECT_LE(count_fds(), before + 1);
}
#endif
