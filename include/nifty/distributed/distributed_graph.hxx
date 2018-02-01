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
        typedef uint64_t NodeType;
        typedef int64_t EdgeType;
        // NodeAdjacency: maps nodes that are adjacent to a given node to the corresponding
        // edge-id
        typedef std::unordered_map<NodeType, EdgeType> NodeAdjacency;
        // NodeStorage: storage of the adjacency for all nodes
        typedef std::unordered_map<NodeType, NodeAdjacency> NodeStorage;
        // TODO do we need dense or non-dense edge storage
        // if we keep it dense, we need some global to local conversion ?!
        // EdgeStorage
        typedef std::pair<NodeType, NodeType> Edge;
        typedef std::vector<Edge> EdgeStorage;
    public:

        // TODO API
        Graph();

    private:
        NodeStorage nodes_;
        EdgeStorage edges_;
    };



}
}
