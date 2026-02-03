#include "basalt/io/file_mapping.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>

using namespace basalt::io;

class FileMappingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary file
        filename = "test_map_file.bin";
        std::ofstream outfile(filename, std::ios::binary);
        const char* msg = "Hello from Memory Map!";
        file_size = std::strlen(msg);
        outfile.write(msg, file_size);
        outfile.close();
    }

    void TearDown() override {
        std::remove(filename.c_str());
    }

    std::string filename;
    size_t file_size;
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
    EXPECT_NE(mapping2.data(), nullptr); // new should be valid
    
    const char* data = static_cast<const char*>(mapping2.data());
    std::string content(data, file_size);
    EXPECT_EQ(content, "Hello from Memory Map!");
}

TEST_F(FileMappingTest, FileNotFound) {
    EXPECT_THROW({
        FileMapping mapping("non_existent_file.bin");
    }, std::exception);
}
