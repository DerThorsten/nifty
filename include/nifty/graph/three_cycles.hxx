#pragma once
#ifndef NIFTY_GRAPH_THREE_CYCLES_HXX
#define NIFTY_GRAPH_THREE_CYCLES_HXX

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/detail/search_impl.hxx"

namespace nifty{
namespace graph{

    template<class GRAPH>
    void findThreeCyclesEdges(
        const GRAPH & graph,
        std::vector< std::array<uint64_t, 3 > > & threeCycles
    ){
        typedef std::array<uint64_t, 3> ThreeCycleEdges;
        threeCycles.clear();

        for(auto u : graph.nodes() ){
            for(auto av : graph.adjacency(u)){
                const auto v = av.node();
                if(u<v){
                    for(auto aw : graph.adjacency(v)){
                        const auto w = aw.node();
                        if(v < w){
                            for(auto auu : graph.adjacency(w)){
                                const auto uu = auu.node();
                                if(uu == u){
                                    ThreeCycleEdges tce{{
                                                            static_cast<uint64_t>(av.edge()),
                                                            static_cast<uint64_t>(aw.edge()),
                                                            static_cast<uint64_t>(auu.edge())
                                                        }};
                                    threeCycles.push_back(tce);
                                    goto exitNestedLoop;
                                }
                            }
                        }
                    }
                }   
            }
            exitNestedLoop:
            ;
        }
    }

    template<class GRAPH>
    std::vector< std::array<uint64_t, 3 > > 
    findThreeCyclesEdges(
        const GRAPH & graph
    ){
        std::vector< std::array<uint64_t, 3 > >  threeCycles;
        findThreeCyclesEdges(graph, threeCycles);
        return threeCycles;
    }

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_THREE_CYCLES_HXX
