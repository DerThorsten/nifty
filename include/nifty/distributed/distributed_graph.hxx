#pragma once

#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "nifty/distributed/region_graph.hxx"


namespace nifty {
namespace distributed {

    // A simple directed graph,
    // that can be constructed from the distributed region graph outputs
    // We use this instead of the nifty graph api, because we need to support
    // non-dense node indices
    class Graph {
        // private graph typedefs

        // NodeAdjacency: maps nodes that are adjacent to a given node to the corresponding edge-id
        typedef std::unordered_map<NodeType, EdgeIndexType> NodeAdjacency;
        // NodeStorage: storage of the adjacency for all nodes
        typedef std::unordered_map<NodeType, NodeAdjacency> NodeStorage;
        // EdgeStorage: dense storage of pairs of edges
        typedef std::vector<EdgeType> EdgeStorage;
    public:

        // API: we can construct the graph from blocks that were extracted via `extractGraphFromRoi`
        // or `mergeSubgraphs` from `region_graph.hxx`
        // TODO we should support extraction from multiple ROI's at some point

        Graph(const std::string & blockPath) {
            loadEdges(blockPath, edges_);
            initGraph();
        }

        // non-constructor API

        // Find edge-id corresponding to the nodes u, v
        // returns -1 if no such edge exists
        EdgeType findEdge(NodeType u, NodeType v) const {
            // find the node iterator
            auto uIt = nodes_.find(u);
            // don't find the u node -> return -1
            if(uIt == nodes_.end()) {
                return -1;
            }
            auto vIt = uIt->find(v);
            // v node is not in u's adjacency -> return -1
            if(vIt == uIt->end()) {
                return -1;
            }
            // otherwise we have found the edge and return the edge id
            return uIt->second();
        }

        // number of nodes and edges
        size_t numberOfNodes() const {return NodeStorage.size();}
        size_t numberOfEdges() const {return EdgeStorage.size();}

    private:
        // init the graph from the edges
        void initGraph() {
            // iterate over the edges we have
            NodeType u, v;
            EdgeIdexType edgeId;
            for(const auto & edge : edges_) {
                u = edge.first;
                v = edge.second;

                // insert v in the u adjacency
                auto uIt = nodes_.find(u);
                if(uIt == nodes_.end()) {
                    // if u is not in the nodes vector yet, insert it
                    uIt = nodes_.insert(std::make_pair(u, NodeStorage()));
                }
                *(uIt)[v] = edgeId;

                // insert u in the v adjacency
                auto vIt = nodes_.find(v);
                if(vIt == nodes_.end()) {
                    // if v is not in the nodes vector yet, insert it
                    vIt = nodes_.insert(std::make_pair(v, NodeStorage()));
                }
                *(vIt)[u] = edgeId;

                // increase the edge id
                ++edgeId;
            }
        }

        NodeStorage nodes_;
        EdgeStorage edges_;
    };



}
}
