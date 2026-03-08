#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path benchmark_executable_path() {
#if defined(BASALT_BENCH_PATH)
    return std::filesystem::path(BASALT_BENCH_PATH);
#elif defined(_WIN32)
    return std::filesystem::current_path() / "basalt_bench.exe";
#else
    return std::filesystem::current_path() / "basalt_bench";
#endif
}

std::string load_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::string text;
    file.seekg(0, std::ios::end);
    text.resize(static_cast<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(text.data(), static_cast<std::streamsize>(text.size()));
    return text;
}

bool contains_array_field(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\": [";
    return json.find(needle) != std::string::npos;
}

} // namespace

TEST(BenchmarkContractTest, JsonStatsUseArrays) {
    const std::filesystem::path exe = benchmark_executable_path();
    ASSERT_TRUE(std::filesystem::exists(exe)) << "Missing benchmark executable: " << exe.string();

    const std::filesystem::path output = std::filesystem::current_path() / "benchmark_contract_output.json";
    std::filesystem::remove(output);

    std::string command;
#if defined(_WIN32)
    command =
        "cmd /c \"\"" + exe.string() + "\" --sizes 256 --threads 2 --simd avx2 --numa off "
        "--memory-policy bind-worker-buffers --shots 1 --warmup 0 --json-out \"" +
        output.string() + "\"\"";
#else
    command =
        "\"" + exe.string() + "\" --sizes 256 --threads 2 --simd avx2 --numa off "
        "--memory-policy bind-worker-buffers --shots 1 --warmup 0 --json-out \"" +
        output.string() + "\"";
#endif
    ASSERT_EQ(std::system(command.c_str()), 0) << command;
    ASSERT_TRUE(std::filesystem::exists(output));

    const std::string json = load_text(output);
    EXPECT_NE(json.find("\"results\": ["), std::string::npos);
    EXPECT_TRUE(contains_array_field(json, "preferred_node_counts"));
    EXPECT_TRUE(contains_array_field(json, "worker_home_node_counts"));
    EXPECT_TRUE(contains_array_field(json, "local_queue_executions_per_node"));
    EXPECT_TRUE(contains_array_field(json, "remote_steals_from_node"));
}
