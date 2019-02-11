#pragma once

#include <boost/pending/disjoint_sets.hpp>
#include "nifty/xtensor/xtensor.hxx"
#include "nifty/graph/undirected_list_graph.hxx"


namespace nifty {
namespace graph {


    // we provide implementations with kruskal and prim
    template<class GRAPH, class EDGES>
    class CarvingSegmenter {
        public:

            // TODO for now I made sorting edges optional, we could also just always
            // do this, but I still want to check how much time this costs
            CarvingSegmenter(const GRAPH & graph,
                             const EDGES & edgeWeights,
                             const bool sortEdges) : graph_(graph),
                                                     edgeWeights_(edgeWeights),
                                                     nNodes_(graph.numberOfNodes()){
                // check that the number of edges and len of edges agree
                NIFTY_CHECK_OP(edgeWeights_.size(), ==, graph_.numberOfEdges(), "Number of edges does not agree");
                if(sortEdges) {
                    sortEdgeIndices();
                }
            }

            template<class NODES>
            inline void operator()(NODES & seeds,
                                   const double bias,
                                   const double noBiasBelow) const {
                // check that the number of nodes agree
                NIFTY_CHECK_OP(seeds.size(), ==, nNodes_, "Number of nodes does not agree");

                // check if we can use kruskal: we don't have a bias and edges were pre-sorted
                const bool useKruskal = (bias == 1.) && (edgesSorted_.size() == graph_.numberOfEdges());
                if(useKruskal) {
                    runKruskal(seeds);
                }
                // otherwise we need to run prim
                else {
                    runPrim(seeds, bias, noBiasBelow);
                }

            }

            inline std::size_t nNodes() const {
                return graph_.numberOfNodes();
            }

            inline const std::vector<std::size_t> & edgesSorted() const {
                return edgesSorted_;
            }
        private:
            // argsort the edges
            inline void sortEdgeIndices() {
                // we sort edge indices in ascending order
                edgesSorted_.resize(graph_.numberOfEdges());
                std::iota(edgesSorted_.begin(), edgesSorted_.end(), 0);
                std::sort(edgesSorted_.begin(), edgesSorted_.end(), [&](const std::size_t a,
                                                                        const std::size_t b){
                    return edgeWeights_[a] < edgeWeights_[b];}
                );
            }

            template<class NODES>
            inline void runPrim(NODES & seeds,
                                const double bias,
                                const double noBiasBelow) const {
                typedef typename NODES::value_type NodeType;
                typedef typename EDGES::value_type WeightType;
                const NodeType backgroundSeedLabel = 1;

                // initialize the priority queue
                typedef std::pair<std::size_t, WeightType> PQElement; // PQElement contains the edge-id and the weight
                auto pqCompare = [](PQElement left, PQElement right) {return left.second < right.second;};
                typedef std::priority_queue<PQElement, std::vector<PQElement>, decltype(pqCompare)> PriorityQueue;
                PriorityQueue pq(pqCompare);

                // TODO for this it would be more efficient to get sparse seeds
                // put edges from seed nodes on the pq
                for(std::size_t nodeId = 0; nodeId < nNodes_; ++nodeId) {
                    const NodeType seedId = seeds[nodeId];
                    if(seedId != 0) {

                        // check if this is a background seed and we use bias
                        const bool needBias = seedId == backgroundSeedLabel;

                        // iterate over the edges going from this node
                        // and put them on the pq
                        for(auto adjIt = graph_.adjacencyBegin(nodeId); adjIt != graph_.adjacencyEnd(nodeId); ++adjIt) {
                            const std::size_t edgeId = adjIt->edge();
                            WeightType weight = edgeWeights_[edgeId];
                            if(needBias && weight > noBiasBelow) {
                                weight *= bias;
                            }

                            pq.push(std::make_pair(edgeId, weight));
                        }
                    }
                }

                // run prim
                while(!pq.empty()) {
                    // extract next element from the queue
                    const PQElement elem = pq.top();
                    pq.pop();

                    const std::size_t edgeId = elem.first;

                    const auto u = graph_.u(edgeId);
                    const auto v = graph_.v(edgeId);
                    const NodeType lU = seeds[u];
                    const NodeType lV = seeds[v];

                    // check for seeds
                    if(lU == 0 && lV == 0){
                        throw std::runtime_error("both have no labels");
                    }
                    else if(lU != 0 && lV != 0){
                        continue;
                    }

                    const auto unlabeledNode = lU == 0 ? u : v;
                    const NodeType seedId = lU == 0 ? lV : lU;

                    // assign seedId to unlabeled node
                    seeds[unlabeledNode] = seedId;

                    // check if this is a background seed and we use bias
                    const bool needBias = seedId == backgroundSeedLabel;

                    // put outgoing edges on the pq
                    for(auto adjIt = graph_.adjacencyBegin(unlabeledNode); adjIt != graph_.adjacencyEnd(unlabeledNode); ++adjIt) {
                        const std::size_t nextEdge = adjIt->edge();

                        // check that this is not the ingoing edge
                        if(nextEdge == edgeId) {
                            continue;
                        }
                        // check that the node is not labeled
                        const auto nextNode = adjIt->node();
                        if(seeds[nextNode] != 0) {
                            continue;
                        }

                        WeightType weight = edgeWeights_[nextEdge];
                        if(needBias && weight > noBiasBelow) {
                            weight *= bias;
                        }
                        pq.push(std::make_pair(nextEdge, weight));
                    }

                }
            }

            template<class NODES>
            inline void runKruskal(NODES & seeds) const {
                typedef typename NODES::value_type NodeType;
                // make union find and map seeds to reperesentatives
                std::vector<uint64_t> ranks(nNodes_);
                std::vector<uint64_t> parents(nNodes_);
                boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
                for(uint64_t node = 0; node < nNodes_; ++node) {
                    ufd.make_set(node);
                }

                // run kruskal
                for(const std::size_t edgeId : edgesSorted_) {
                    // get the nodes connected by this edge
                    // and the representatives
                    const uint64_t u = graph_.u(edgeId);
                    const uint64_t v = graph_.v(edgeId);
                    const uint64_t ru = ufd.find_set(u);
                    const uint64_t rv = ufd.find_set(v);

                    // if the representatives are the same, continue
                    if(ru == rv) {
                        continue;
                    }

                    // get the seeds for our reperesentatives
                    const NodeType lu = seeds[ru];
                    const NodeType lv = seeds[rv];

                    // if we have two seeded regions (both values different from 0) continue
                    if(lu !=0 && lv != 0) {
                        continue;
                    }

                    // otherwise link the two representatives
                    ufd.link(ru, rv);

                    // if we have a seed, propagate it
                    if(lu != 0) {
                        seeds[rv] = lu;
                    }
                    if(lv != 0) {
                        seeds[ru] = lv;
                    }

                }
            }

        private:
            const GRAPH & graph_;
            const EDGES & edgeWeights_;
            std::size_t nNodes_;
            std::vector<std::size_t> edgesSorted_;
    };


}
}
