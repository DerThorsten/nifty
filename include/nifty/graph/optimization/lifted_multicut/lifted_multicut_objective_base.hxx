#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_BASE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_BASE_HXX



#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{





    template<class CHILD_OBJECTIVE, class GRAPH, class LIFTED_GRAPH, class WEIGHT_TYPE>
    class LiftedMulticutObjectiveBase{
    public:

        typedef CHILD_OBJECTIVE ChildObjective;
        typedef LiftedMulticutObjectiveBase<ChildObjective, GRAPH, LIFTED_GRAPH, WEIGHT_TYPE> Self;
    

        template<class NODE_LABELS>
        WEIGHT_TYPE evalNodeLabels(const NODE_LABELS & nodeLabels)const{
            WEIGHT_TYPE sum = static_cast<WEIGHT_TYPE>(0.0);

            const auto & w = _child().weights();
            const auto & lg = _child().liftedGraph();

            for(const auto edge: lg.edges()){
                const auto uv = lg.uv(edge);

                if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                    sum += w[edge];
                }
            }
            return sum;
        }

        uint64_t numberOfLiftedEdges()const{
            return _child().liftedGraph().numberOfEdges() - _child().graph().numberOfEdges();
        }


    private:
        ChildObjective & _child(){
           return *static_cast<ChildObjective *>(this);
        }
        const ChildObjective & _child()const{
           return *static_cast<const ChildObjective *>(this);
        }

    };


} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_BASE_HXX
