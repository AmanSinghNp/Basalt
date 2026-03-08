#include "basalt/internal/numa_topology.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#if defined(__linux__)
#include <filesystem>
#endif

namespace basalt::internal {

namespace {

#if defined(__linux__)
std::vector<int> parse_cpu_list(const std::string& text) {
    std::vector<int> cpus;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const size_t dash = token.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(std::stoi(token));
            continue;
        }

        const int start = std::stoi(token.substr(0, dash));
        const int end = std::stoi(token.substr(dash + 1));
        for (int cpu = start; cpu <= end; ++cpu) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    int value = 0;
    while (stream >> value) {
        values.push_back(value);
    }
    return values;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        return {};
    }
    std::string value;
    std::getline(file, value);
    return value;
}
#endif

} // namespace

NumaTopology NumaTopology::detect() {
    NumaTopology topology;

#if defined(__linux__)
    namespace fs = std::filesystem;
    const fs::path node_root("/sys/devices/system/node");
    if (!fs::exists(node_root) || !fs::is_directory(node_root)) {
        topology.m_available = false;
        topology.m_is_uma = true;
        topology.m_nodes.push_back(NumaNodeInfo{0, {}, {}});
        return topology;
    }

    std::vector<fs::path> node_paths;
    for (const fs::directory_entry& entry : fs::directory_iterator(node_root)) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("node", 0) != 0 || name.size() <= 4) {
            continue;
        }
        node_paths.push_back(entry.path());
    }

    std::sort(node_paths.begin(), node_paths.end());
    if (node_paths.empty()) {
        topology.m_available = false;
        topology.m_is_uma = true;
        topology.m_nodes.push_back(NumaNodeInfo{0, {}, {}});
        return topology;
    }

    topology.m_available = true;
    topology.m_nodes.reserve(node_paths.size());
    int max_cpu = -1;

    for (const fs::path& path : node_paths) {
        NumaNodeInfo node;
        node.id = static_cast<size_t>(std::stoul(path.filename().string().substr(4)));
        node.cpus = parse_cpu_list(read_text_file(path / "cpulist"));
        node.distances = parse_int_list(read_text_file(path / "distance"));
        for (int cpu : node.cpus) {
            max_cpu = std::max(max_cpu, cpu);
        }
        topology.m_nodes.push_back(std::move(node));
    }

    topology.m_cpu_to_node.assign(static_cast<size_t>(max_cpu + 1), 0);
    for (const NumaNodeInfo& node : topology.m_nodes) {
        for (int cpu : node.cpus) {
            topology.m_cpu_to_node[static_cast<size_t>(cpu)] = static_cast<int>(node.id);
        }
    }
    topology.m_is_uma = topology.m_nodes.size() <= 1;
#else
    topology.m_available = false;
    topology.m_is_uma = true;
    NumaNodeInfo node;
    node.id = 0;
    const unsigned concurrency = std::thread::hardware_concurrency();
    for (unsigned cpu = 0; cpu < concurrency; ++cpu) {
        node.cpus.push_back(static_cast<int>(cpu));
    }
    topology.m_nodes.push_back(std::move(node));
#endif

    return topology;
}

size_t NumaTopology::node_for_cpu(int cpu) const noexcept {
    if (cpu < 0 || static_cast<size_t>(cpu) >= m_cpu_to_node.size()) {
        return 0;
    }
    return static_cast<size_t>(m_cpu_to_node[static_cast<size_t>(cpu)]);
}

} // namespace basalt::internal
