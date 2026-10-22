#include <gtest/gtest.h>

#ifdef BASALT_BENCH_PATH

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

std::filesystem::path benchmark_executable_path() {
    return std::filesystem::path(BASALT_BENCH_PATH);
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

namespace {

// Runs basalt_bench with the given arguments (output discarded) and returns its exit status.
int run_bench_exit_code(const std::string& args) {
#if defined(_WIN32)
    const std::string null_device = "NUL";
#else
    const std::string null_device = "/dev/null";
#endif
    const std::string command =
        "\"" + benchmark_executable_path().string() + "\" " + args + " > " + null_device + " 2>&1";
    const int status = std::system(command.c_str());
#if defined(_WIN32)
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

} // namespace

TEST(BenchmarkContractTest, MalformedArgumentsExitWithCode2) {
    EXPECT_EQ(run_bench_exit_code("--shots abc"), 2);
    EXPECT_EQ(run_bench_exit_code("--shots 99999999999999999999999"), 2);
    EXPECT_EQ(run_bench_exit_code("--seed -1"), 2);
    EXPECT_EQ(run_bench_exit_code("--sizes 256,x"), 2);
    EXPECT_EQ(run_bench_exit_code("--bogus"), 2);
}

TEST(BenchmarkContractTest, HelpExitsZero) {
    EXPECT_EQ(run_bench_exit_code("--help"), 0);
}

#else

TEST(BenchmarkContractTest, JsonStatsUseArrays) {
    GTEST_SKIP() << "basalt_bench is not configured (BASALT_BENCH_PATH is not defined); benchmarks are disabled";
}

#endif // BASALT_BENCH_PATH
