#pragma once
#ifndef NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class CHILD_OBJECTIVE, class GRAPH, class LIFTED_GRAPH, class WEIGHT_TYPE>
    class LiftedMulticutObjectiveBase{
    public:

        typedef CHILD_OBJECTIVE ChildObjective;
        typedef LiftedMulticutObjectiveBase<ChildObjective, GRAPH, LIFTED_GRAPH, WEIGHT_TYPE> Self;


    };


    template<class GRAPH, class WEIGHT_TYPE>   
    class LiftedMulticutObjective :  public
        LiftedMulticutObjectiveBase<
            LiftedMulticutObjective<GRAPH, WEIGHT_TYPE>, 
            GRAPH, UndirectedGraph<>, WEIGHT_TYPE
        >
    {   
    public:

        // static_assert(std::is_same<GRAPH::NodeIdTag, ContiguousTag>::value
        //     "Currently only graphs with contiguous node ids are supported"
        // );
        // static_assert(std::is_same<GRAPH::EdgeIdTag, ContiguousTag>::value
        //     "Currently only graphs with contiguous edge ids are supported"
        // );
        // static_assert(std::is_same<GRAPH::NodeIdOrderTag, SortedTag>::value
        //     "Currently only graphs with contiguous node ids are supported"
        // );
        // static_assert(std::is_same<GRAPH::EdgeIdOrderTag, SortedTag>::value
        //     "Currently only graphs with contiguous edge ids are supported"
        // );

        typedef GRAPH Graph;
        typedef UndirectedGraph<> LiftedGraph;
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<LiftedGraph, WeightType> WeightsMap;
        
        LiftedMulticutObjective(const Graph & graph, const int64_t reserveAdditionalEdges = -1)
        :   graph_(graph),
            liftedGraph_(graph.numberOfNodes(), graph.numberOfEdges() + (reserveAdditionalEdges<0 ?  graph.numberOfEdges() : reserveAdditionalEdges)),
            weights_(liftedGraph_){




            for(const auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                liftedGraph_.insertEdge(uv.first, uv.second);
            }
        }

        void setCost(const uint64_t u, const uint64_t v, const WeightType & w){
            // check if that is a new edge or not
        }

    private:
        const Graph & graph_;
        LiftedGraph liftedGraph_;
        WeightsMap weights_;
    };

} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
