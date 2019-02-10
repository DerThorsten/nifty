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

            template<class EDGES>
            CarvingSegmenter(const GRAPH & graph,
                             const EDGES & edges,
                             const bool fromSerialization) : graph_(graph),
                                                             nNodes_(graph.numberOfNodes()){
                // check that the number of edges and len of edges agree
                NIFTY_CHECK_OP(edges.size(), ==, graph_.numberOfEdges(), "Number of edges does not agree");

                edgesSorted_.resize(graph_.numberOfEdges());
                // just copy th sorted edge indices if we load from serialization
                if(fromSerialization) {
                    std::copy(edges.begin(), edges.end(), edgesSorted_.begin());
                // otherwise, sort the edge indices based on the edge weights
                } else {
                    // argsort the edges
                    // we sort in ascending order TODO is this correct ?
                    std::iota(edgesSorted_.begin(), edgesSorted_.end(), 0);
                    std::sort(edgesSorted_.begin(), edgesSorted_.end(), [&](const std::size_t a,
                                                                            const std::size_t b){
                        return edges[a] < edges[b];}
                    );
                }
            }

            // TODO is this the best way to implement kruskal for seeded ws ?
            template<class NODES>
            inline void operator()(const NODES & seeds,
                                   NODES & nodeLabeling) const {
                typedef typename NODES::value_type NodeType;
                // check that the number of nodes agree
                NIFTY_CHECK_OP(seeds.size(), ==, nNodes_, "Number of nodes does not agree");
                NIFTY_CHECK_OP(nodeLabeling.size(), ==, nNodes_, "Number of nodes does not agree");

                // make union find and map seeds to reperesentatives
                std::vector<uint64_t> ranks(nNodes_);
                std::vector<uint64_t> parents(nNodes_);
                boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
                for(uint64_t node = 0; node < nNodes_; ++node) {
                    ufd.make_set(node);
                    // TODO do we need to call find_set on node here? I hope that disjoint seeds
                    // does the reasonable thing and assigns node the representative node here ...
                    // nodeLabeling[ufd.find_set(node)] = seeds[node];
                    nodeLabeling[node] = seeds[node];
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
                    const NodeType lu = nodeLabeling[ru];
                    const NodeType lv = nodeLabeling[rv];

                    // if we have two seeded regions (both values different from 0) continue
                    if(lu !=0 && lv != 0) {
                        continue;
                    }

                    // otherwise link the two representatives
                    ufd.link(ru, rv);

                    // if we have a seed, propagate it
                    if(lu != 0) {
                        nodeLabeling[rv] = lu;
                    }
                    if(lv != 0) {
                        nodeLabeling[ru] = lv;
                    }

                }

                // write final node labeling
                for(std::size_t node = 0; node < nNodes_; ++node) {
                    nodeLabeling[node] = nodeLabeling[ufd.find_set(node)];
                }
            }

            inline std::size_t nNodes() const {
                return graph_.numberOfNodes();
            }

            inline const std::vector<std::size_t> & edgesSorted() const {
                return edgesSorted_;
            }

        private:
            const GRAPH & graph_;
            std::size_t nNodes_;
            std::vector<std::size_t> edgesSorted_;
    };


}
}
