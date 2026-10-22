#include "basalt/internal/numa_topology.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

namespace basalt::internal {

namespace {

#if defined(__linux__)

// Sanity limits so that corrupt sysfs content cannot trigger huge allocations.
constexpr long kMaxCpuNumber = 1 << 16;
constexpr unsigned long kMaxNodeId = 1 << 16;

bool parse_long(std::string_view text, long& out) {
    if (text.empty()) {
        return false;
    }
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, out);
    return result.ec == std::errc() && result.ptr == last;
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' ||
                             text.front() == '\n' || text.front() == '\r')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' ||
                             text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    return text;
}

// Parses "0-3,8,10-11". An empty string is a valid empty list (memory-only node).
// Returns nullopt on malformed input.
std::optional<std::vector<int>> parse_cpu_list(std::string_view text) {
    std::vector<int> cpus;
    text = trim(text);
    while (!text.empty()) {
        const size_t comma = text.find(',');
        const std::string_view token =
            trim(comma == std::string_view::npos ? text : text.substr(0, comma));
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);

        long start = 0;
        long end = 0;
        const size_t dash = token.find('-');
        if (dash == std::string_view::npos) {
            if (!parse_long(token, start)) {
                return std::nullopt;
            }
            end = start;
        } else if (!parse_long(token.substr(0, dash), start) ||
                   !parse_long(token.substr(dash + 1), end)) {
            return std::nullopt;
        }
        if (start < 0 || end < start || end >= kMaxCpuNumber) {
            return std::nullopt;
        }
        for (long cpu = start; cpu <= end; ++cpu) {
            cpus.push_back(static_cast<int>(cpu));
        }
    }
    return cpus;
}

// Distances are advisory; stop at the first token that is not an integer.
std::vector<int> parse_int_list(std::string_view text) {
    std::vector<int> values;
    text = trim(text);
    while (!text.empty()) {
        const size_t space = text.find_first_of(" \t");
        const std::string_view token = text.substr(0, space);
        long value = 0;
        if (!parse_long(token, value) || value < 0 || value > 1000000) {
            break;
        }
        values.push_back(static_cast<int>(value));
        if (space == std::string_view::npos) {
            break;
        }
        text = trim(text.substr(space));
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

// Returns the numeric id for a directory named "node<digits>", or nullopt.
std::optional<size_t> parse_node_name(const std::string& name) {
    if (name.size() <= 4 || name.compare(0, 4, "node") != 0) {
        return std::nullopt;
    }
    unsigned long id = 0;
    const char* first = name.data() + 4;
    const char* last = name.data() + name.size();
    const auto result = std::from_chars(first, last, id);
    if (result.ec != std::errc() || result.ptr != last || id >= kMaxNodeId) {
        return std::nullopt;
    }
    return static_cast<size_t>(id);
}

struct ParsedTopology {
    std::vector<NumaNodeInfo> nodes;
    std::vector<int> cpu_to_node;
};

// Returns nullopt when the tree is absent, empty or malformed (caller falls back to UMA).
std::optional<ParsedTopology> parse_sysfs(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    std::error_code ec;

    if (!fs::is_directory(root, ec) || ec) {
        return std::nullopt;
    }

    fs::directory_iterator it(root, ec);
    if (ec) {
        return std::nullopt;
    }

    std::vector<std::pair<size_t, fs::path>> node_paths;
    for (const fs::directory_iterator end; it != end; it.increment(ec)) {
        if (ec) {
            return std::nullopt;
        }
        std::error_code entry_ec;
        if (!it->is_directory(entry_ec) || entry_ec) {
            continue;
        }
        const std::optional<size_t> id = parse_node_name(it->path().filename().string());
        if (!id) {
            continue;
        }
        node_paths.emplace_back(*id, it->path());
    }
    if (ec || node_paths.empty()) {
        return std::nullopt;
    }

    // Numeric order: "node2" must come before "node10".
    std::sort(node_paths.begin(), node_paths.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t i = 1; i < node_paths.size(); ++i) {
        if (node_paths[i].first == node_paths[i - 1].first) {
            return std::nullopt;
        }
    }

    ParsedTopology parsed;
    parsed.nodes.reserve(node_paths.size());
    int max_cpu = -1;
    for (const auto& [id, path] : node_paths) {
        auto cpus = parse_cpu_list(read_text_file(path / "cpulist"));
        if (!cpus) {
            return std::nullopt;
        }
        NumaNodeInfo node;
        node.id = id;
        node.cpus = std::move(*cpus);
        node.distances = parse_int_list(read_text_file(path / "distance"));
        for (int cpu : node.cpus) {
            max_cpu = std::max(max_cpu, cpu);
        }
        parsed.nodes.push_back(std::move(node));
    }

    parsed.cpu_to_node.assign(static_cast<size_t>(max_cpu + 1), -1);
    for (const NumaNodeInfo& node : parsed.nodes) {
        for (int cpu : node.cpus) {
            parsed.cpu_to_node[static_cast<size_t>(cpu)] = static_cast<int>(node.id);
        }
    }
    return parsed;
}

#endif

} // namespace

NumaTopology NumaTopology::detect() {
    return detect(std::filesystem::path("/sys/devices/system/node"));
}

NumaTopology NumaTopology::detect(const std::filesystem::path& root) {
    NumaTopology topology;
    topology.m_available = false;
    topology.m_is_uma = true;

#if defined(__linux__)
    try {
        std::optional<ParsedTopology> parsed = parse_sysfs(root);
        if (parsed) {
            topology.m_available = true;
            topology.m_nodes = std::move(parsed->nodes);
            topology.m_cpu_to_node = std::move(parsed->cpu_to_node);
            topology.m_is_uma = topology.m_nodes.size() <= 1;
            return topology;
        }
    } catch (...) {
        // Any failure (including allocation) falls through to the UMA topology below.
        topology = NumaTopology();
    }
    topology.m_nodes.push_back(NumaNodeInfo{0, {}, {}});
#else
    (void)root;
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

size_t NumaTopology::index_for_node_id(size_t id) const noexcept {
    const auto it = std::lower_bound(
        m_nodes.begin(), m_nodes.end(), id,
        [](const NumaNodeInfo& node, size_t value) { return node.id < value; });
    if (it == m_nodes.end() || it->id != id) {
        return npos;
    }
    return static_cast<size_t>(it - m_nodes.begin());
}

size_t NumaTopology::node_id_for_index(size_t index) const noexcept {
    if (m_nodes.empty()) {
        return 0;
    }
    return m_nodes[index % m_nodes.size()].id;
}

size_t NumaTopology::node_for_cpu(int cpu) const noexcept {
    if (cpu >= 0 && static_cast<size_t>(cpu) < m_cpu_to_node.size() &&
        m_cpu_to_node[static_cast<size_t>(cpu)] >= 0) {
        return static_cast<size_t>(m_cpu_to_node[static_cast<size_t>(cpu)]);
    }
    return node_id_for_index(0);
}

size_t NumaTopology::node_index_for_cpu(int cpu) const noexcept {
    const size_t index = index_for_node_id(node_for_cpu(cpu));
    return index == npos ? 0 : index;
}

} // namespace basalt::internal
