#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace basalt::internal {

struct NumaNodeInfo {
    size_t id = 0;
    std::vector<int> cpus;
    std::vector<int> distances;
};

/// Snapshot of the machine's NUMA layout.
///
/// Terminology: a node *id* is the kernel's identifier (the N in /sys/devices/system/node/nodeN)
/// and may be sparse (node0, node2) or larger than the node count. A node *index* is the dense
/// position 0..node_count()-1 in nodes(), which is sorted by ascending numeric id. Containers
/// sized by node_count() must be indexed by index; anything handed to the kernel (mbind) must
/// use the id.
class NumaTopology {
public:
    static constexpr size_t npos = static_cast<size_t>(-1);

    /// Detects the topology from /sys/devices/system/node. Never throws: on any failure
    /// (missing or unreadable sysfs, malformed content) a single-node UMA topology is returned.
    static NumaTopology detect();

    /// Same as detect() but reads the node directories from `root` (for tests).
    static NumaTopology detect(const std::filesystem::path& root);

    [[nodiscard]] bool is_uma() const noexcept { return m_is_uma; }
    [[nodiscard]] bool available() const noexcept { return m_available; }
    [[nodiscard]] size_t node_count() const noexcept { return m_nodes.size(); }
    /// Nodes sorted by ascending numeric id.
    [[nodiscard]] const std::vector<NumaNodeInfo>& nodes() const noexcept { return m_nodes; }
    /// Indexed by cpu number; holds the node id of that cpu, or -1 for cpus not listed by any node.
    [[nodiscard]] const std::vector<int>& cpu_to_node() const noexcept { return m_cpu_to_node; }

    /// Node id owning `cpu`; the first node's id when the cpu is unknown.
    [[nodiscard]] size_t node_for_cpu(int cpu) const noexcept;
    /// Dense index (into nodes()) of the node owning `cpu`; 0 when the cpu is unknown.
    [[nodiscard]] size_t node_index_for_cpu(int cpu) const noexcept;
    /// Dense index of the node with kernel id `id`, or npos if there is no such node.
    [[nodiscard]] size_t index_for_node_id(size_t id) const noexcept;
    /// Kernel id of the node at dense `index` (taken modulo node_count()); 0 for an empty topology.
    [[nodiscard]] size_t node_id_for_index(size_t index) const noexcept;

private:
    bool m_available = false;
    bool m_is_uma = true;
    std::vector<NumaNodeInfo> m_nodes;
    std::vector<int> m_cpu_to_node;
};

} // namespace basalt::internal
