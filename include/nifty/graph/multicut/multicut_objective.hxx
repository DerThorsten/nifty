#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_OBJECTIVE_HXX

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"

namespace nifty{
namespace graph{


    template<class CHILD_OBJECTIVE, class GRAPH, class WEIGHT_TYPE>
    class MulticutObjectiveBase{
    public:

    typedef CHILD_OBJECTIVE ChildObjective;
    typedef MulticutObjectiveBase<ChildObjective, GRAPH, WEIGHT_TYPE> Self;

    private:

    };


    template<class GRAPH, class WEIGHT_TYPE>   
    class MulticutObjective :  public
        MulticutObjectiveBase<
            MulticutObjective<GRAPH, WEIGHT_TYPE>, GRAPH, WEIGHT_TYPE
        >
    {   
    public:
        typedef GRAPH Graph;
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<Graph, WeightType> WeightsMap;
        MulticutObjective(const Graph & g )
        :   graph_(g),
            weights_(g, 0.0)
        {

        }
        WeightsMap & weights(){
            return weights_;
        }

        // MUST IMPL INTERFACE
        const Graph & graph() const{
            return graph_;
        }
        const WeightsMap & weights() const{
            return weights_;
        }
    private:
        const Graph & graph_;
        WeightsMap weights_;
    };

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_MULTICUT_MULTICUT_OBJECTIVE_HXX
