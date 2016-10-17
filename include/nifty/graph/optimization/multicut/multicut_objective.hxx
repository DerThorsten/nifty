#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_OBJECTIVE_HXX

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"

namespace nifty{
namespace graph{


    template<class CHILD_OBJECTIVE, class GRAPH, class WEIGHT_TYPE>
    class MulticutObjectiveBase{
    public:

        typedef CHILD_OBJECTIVE ChildObjective;
        typedef MulticutObjectiveBase<ChildObjective, GRAPH, WEIGHT_TYPE> Self;

        template<class NODE_LABELS>
        WEIGHT_TYPE evalNodeLabels(const NODE_LABELS & nodeLabels)const{
            WEIGHT_TYPE sum = static_cast<WEIGHT_TYPE>(0.0);
            const auto & w = _child().weights();
            const auto & g = _child().graph();
            for(const auto edge: g.edges()){
                const auto uv = g.uv(edge);
                
                if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                    sum += w[edge];
                }
            }
            return sum;
        }
    private:
        ChildObjective & _child(){
           return *static_cast<ChildObjective *>(this);
        }
        const ChildObjective & _child()const{
           return *static_cast<const ChildObjective *>(this);
        }

    };


    template<class GRAPH, class WEIGHT_TYPE>   
    class MulticutObjective :  public
        MulticutObjectiveBase<
            MulticutObjective<GRAPH, WEIGHT_TYPE>, GRAPH, WEIGHT_TYPE
        >
    {   
    public:
        typedef GRAPH Graph;
        typedef GRAPH GraphType;
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

#endif  // NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_OBJECTIVE_HXX
