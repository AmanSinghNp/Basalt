#pragma once

#include <cstddef>
#include <vector>

namespace basalt::internal {

struct NumaNodeInfo {
    size_t id = 0;
    std::vector<int> cpus;
    std::vector<int> distances;
};

class NumaTopology {
public:
    static NumaTopology detect();

    [[nodiscard]] bool is_uma() const noexcept { return m_is_uma; }
    [[nodiscard]] bool available() const noexcept { return m_available; }
    [[nodiscard]] size_t node_count() const noexcept { return m_nodes.size(); }
    [[nodiscard]] const std::vector<NumaNodeInfo>& nodes() const noexcept { return m_nodes; }
    [[nodiscard]] const std::vector<int>& cpu_to_node() const noexcept { return m_cpu_to_node; }

    [[nodiscard]] size_t node_for_cpu(int cpu) const noexcept;

private:
    bool m_available = false;
    bool m_is_uma = true;
    std::vector<NumaNodeInfo> m_nodes;
    std::vector<int> m_cpu_to_node;
};

} // namespace basalt::internal
