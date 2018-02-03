#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/distributed/graph_extraction.hxx"

namespace nifty {
namespace distributed {


    inline void loadNiftyGraph(const std::string & graphPath,
                               nifty::graph::UndirectedGraph & g,
                               std::unordered_map<NodeType, NodeType> & relabeling,
                               bool relabelNodes=true) {
        std::vector<NodeType> nodes;
        loadNodes(graphPath, nodes, 0);
        const size_t nNodes = nodes.size();

        std::vector<EdgeType> edges;
        loadEdges(graphPath, edges, 0);
        const size_t nEdges = edges.size();

        if(relabelNodes) {
            g.assign(nNodes, nEdges);
            relabeling.resize(nNodes);
            for(size_t ii = 0; ii < nNods; ++ii) {
                relabeling[nodes[ii]] = ii;
            }

            for(const auto & edge : edges) {
                g.insertEdge(relabeling[edge.first], relabeling[edge.second]);
            }

        } else {
            NodeType maxNode = *std::max_element(nodes.begin(), nodes.end());
            g.assign(maxNode + 1, nEdges);

            for(const auto & edge : edges) {
                g.insertEdge(edge.first, edge.second);
            }
        }

    }


}
}
