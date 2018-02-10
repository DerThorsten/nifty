#pragma once

#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "nifty/distributed/graph_extraction.hxx"


namespace nifty {
namespace distributed {

    // A simple directed graph,
    // that can be constructed from the distributed region graph outputs
    // We use this instead of the nifty graph api, because we need to support
    // non-dense node indices
    class Graph {
        // private graph typedefs

        // NodeAdjacency: maps nodes that are adjacent to a given node to the corresponding edge-id
        typedef std::map<NodeType, EdgeIndexType> NodeAdjacency;
        // NodeStorage: storage of the adjacency for all nodes
        typedef std::unordered_map<NodeType, NodeAdjacency> NodeStorage;
        // EdgeStorage: dense storage of pairs of edges
        typedef std::vector<EdgeType> EdgeStorage;
    public:

        // API: we can construct the graph from blocks that were extracted via `extractGraphFromRoi`
        // or `mergeSubgraphs` from `region_graph.hxx`

        Graph(const std::string & blockPath) {
            loadEdges(blockPath, edges_, 0);
            initGraph();
        }

        // This is a bit weird (constructor with side effects....)
        // but I don't want the edge id mapping to be part of this class
        Graph(const std::vector<std::string> & blockPaths,
              std::vector<EdgeIndexType> & edgeIdsOut) {

            // load all the edges and edge-id mapping in the blocks
            // to tmp objects
            std::vector<EdgeType> edgesTmp;
            std::vector<EdgeIndexType> edgeIdsTmp;
            for(const auto & blockPath : blockPaths) {
                loadEdges(blockPath, edgesTmp, edgesTmp.size());
                loadEdgeIndices(blockPath, edgeIdsTmp, edgeIdsTmp.size());
            }

            // get the indices that would sort the edge uv's
            // (we need to sort the edge uvs AND the edgeIds in the same manner here)
            std::vector<size_t> indices(edgesTmp.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](const size_t a, const size_t b){
                return edgesTmp[a] < edgesTmp[b];
            });

            // copy tmp edges in sorted order
            edges_.resize(edgesTmp.size());
            for(size_t ii = 0; ii < edges_.size(); ++ii) {
                edges_[ii] = edgesTmp[indices[ii]];
            }
            // make edges unique
            edges_.resize(std::unique(edges_.begin(), edges_.end()) - edges_.begin());

            // copy tmp edge ids to the out vector in sorted order
            edgeIdsOut.resize(edgeIdsTmp.size());
            for(size_t ii = 0; ii < edgeIdsOut.size(); ++ii) {
                edgeIdsOut[ii] = edgeIdsTmp[indices[ii]];
            }
            // make edge ids unique
            edgeIdsOut.resize(std::unique(edgeIdsOut.begin(), edgeIdsOut.end()) - edgeIdsOut.begin());

            // init the graph
            initGraph();
        }

        // non-constructor API

        // Find edge-id corresponding to the nodes u, v
        // returns -1 if no such edge exists
        EdgeIndexType findEdge(NodeType u, NodeType v) const {
            // find the node iterator
            auto uIt = nodes_.find(u);
            // don't find the u node -> return -1
            if(uIt == nodes_.end()) {
                return -1;
            }
            // check if v is in the adjacency of u
            auto vIt = uIt->second.find(v);
            // v node is not in u's adjacency -> return -1
            if(vIt == uIt->second.end()) {
                return -1;
            }
            // otherwise we have found the edge and return the edge id
            return vIt->second;
        }

        // get the node adjacency
        const NodeAdjacency & nodeAdjacency(const NodeType node) const {
            return nodes_.at(node);
        }

        // number of nodes and edges
        size_t numberOfNodes() const {return nodes_.size();}
        size_t numberOfEdges() const {return edges_.size();}

    private:
        // init the graph from the edges
        void initGraph() {
            // iterate over the edges we have
            NodeType u, v;
            EdgeIndexType edgeId = 0;
            for(const auto & edge : edges_) {
                u = edge.first;
                v = edge.second;

                // insert v in the u adjacency
                auto uIt = nodes_.find(u);
                if(uIt == nodes_.end()) {
                    // if u is not in the nodes vector yet, insert it
                    nodes_.insert(std::make_pair(u, NodeAdjacency{{v, edgeId}}));
                } else {
                    uIt->second[v] = edgeId;
                }

                // insert u in the v adjacency
                auto vIt = nodes_.find(v);
                if(vIt == nodes_.end()) {
                    // if v is not in the nodes vector yet, insert it
                    nodes_.insert(std::make_pair(v, NodeAdjacency{{u, edgeId}}));
                } else {
                    vIt->second[u] = edgeId;
                }

                // increase the edge id
                ++edgeId;
            }
        }

        NodeStorage nodes_;
        EdgeStorage edges_;
    };



}
}
