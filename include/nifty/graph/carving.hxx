#pragma once

#include <boost/pending/disjoint_sets.hpp>
#include "nifty/xtensor/xtensor.hxx"
#include "nifty/graph/undirected_list_graph.hxx"


namespace nifty {
namespace graph {


    // for now implementation without priors, using kruskal
    // (I think priors are actually not used in ilastik, but need to double check !)
    // note that if this is true, we need to sort edges only once, which should
    // yield a major speed-up !
    template<class GRAPH>
    class CarvingSegmenter {
        public:

            template<class EDGE_WEIGHTS>
            CarvingSegmenter(const GRAPH & graph,
                             const EDGE_WEIGHTS & edgeWeights) : graph_(graph),
                                                                 nNodes_(graph.numberOfNodes()){
                // TODO check that the number of edges and len of edgeWeights agree

                // argsort the edges
                // we sort in ascending order TODO is this correct ?
                edgesSorted_.resize(graph_.numberOfEdges());
                std::iota(edgesSorted_.begin(), edgesSorted_.end(), 0);
                std::sort(edgesSorted_.begin(), edgesSorted_.end(), [&](const std::size_t a,
                                                                        const std::size_t b){
                    return edgeWeights[a] < edgeWeights[b];}
                );
            }

            template<class NODE_LABELS>
            inline void operator()(const NODE_LABELS & seeds,
                                   NODE_LABELS & nodeLabeling) const {
                // make union find
                std::vector<uint64_t> ranks(nNodes_);
                std::vector<uint64_t> parents(nNodes_);
                boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
                for(uint64_t node = 0; node < nNodes_; ++node) {
                    ufd.make_set(node);
                }

                // run kruskal TODO check how to do this for seeded ws
                for(const std::size_t edgeId : edgesSorted_) {
                }

                // TODO we need to make sure that we only have two set ids
                // and we need to make sure the mapping background / foreground is still correct
                // write to node labeling
                for(std::size_t node = 0; node < nNodes_; ++node) {
                    nodeLabeling[node] = ufd.find_set(node);
                }
            }

            inline std::size_t nNodes() const {
                return graph_.numberOfNodes();
            }

        private:
            const GRAPH & graph_;
            std::size_t nNodes_;
            std::vector<std::size_t> edgesSorted_;
    };


}
}
