#pragma once


#include <algorithm>
#include <set>

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
        for(auto edge : graph.edges()){
            const auto uv = graph.uv(edge);
            const auto u = uv.first;
            const auto v = uv.second;

            for(auto adj : graph.adjacency(u)){
                const auto w = adj.node();
                const auto secondEdge = adj.edge();
                if(w != v && secondEdge < edge){
                    auto thirdEdge = graph.findEdge(w, v);
                    if(thirdEdge != -1 ){//thirdEdge < secondEdge){
                        
                        //ThreeCycle tc(edge,secondEdge,thirdEdge);
                        //cycleSet.insert(tc);
                        ThreeCycleEdges tce{{
                           static_cast<uint64_t>(edge),
                           static_cast<uint64_t>(secondEdge),
                           static_cast<uint64_t>(thirdEdge)
                        }};
                        threeCycles.push_back(tce);
                    }
                }
            }
        }
        //for(auto c : cycleSet)
        //    threeCycles.push_back(c.edges_);
        //std::cout<<threeCycles.size()<<"\n";
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

