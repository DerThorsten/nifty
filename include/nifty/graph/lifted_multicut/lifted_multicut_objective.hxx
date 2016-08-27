#pragma once
#ifndef NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"
#include "nifty/graph/multicut/multicut_objective.hxx"

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


    private:
        ChildObjective & _child(){
           return *static_cast<ChildObjective *>(this);
        }
        const ChildObjective & _child()const{
           return *static_cast<const ChildObjective *>(this);
        }

    };


    template<class GRAPH, class WEIGHT_TYPE>   
    class LiftedMulticutObjective :  public
        LiftedMulticutObjectiveBase<
            LiftedMulticutObjective<GRAPH, WEIGHT_TYPE>, 
            GRAPH, UndirectedGraph<>, WEIGHT_TYPE
        >
    {   
    private:
        typedef nifty::graph::detail_graph::NodeIndicesToContiguousNodeIndices<GRAPH > ToContiguousNodes;

    public:


        typedef GRAPH Graph;
        typedef UndirectedGraph<> LiftedGraph;
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<LiftedGraph, WeightType> WeightsMap;
        

        LiftedMulticutObjective(const Graph & graph, const int64_t reserveAdditionalEdges = -1)
        :   graph_(graph),
            liftedGraph_(graph.numberOfNodes(), graph.numberOfEdges() + (reserveAdditionalEdges<0 ?  graph.numberOfEdges() : reserveAdditionalEdges)),
            weights_(liftedGraph_),
            toContiguousNodes_(graph_){

            for(const auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                liftedGraph_.insertEdge(
                    toContiguousNodes_[uv.first],
                    toContiguousNodes_[uv.second]
                );
            }
            weights_.insertEdge(liftedGraph_.edgeIdUpperBound(),0);
        }

        void setCost(const uint64_t u, const uint64_t v, const WeightType & w, const bool overwrite = false){
            const auto du = toContiguousNodes_[u];
            const auto dv = toContiguousNodes_[v];
            const auto preSize = liftedGraph_.numberOfEdges();
            const auto edge = liftedGraph_.insertEdge(du,dv);
            if( liftedGraph_.numberOfEdges() > preSize){
                weights_.insertedEdge(edge, w);
            }
            else{
                if(overwrite)
                    weights_[edge] = w;
                else
                    weights_[edge] += w;
            }
        }

        WeightsMap & weights(){
            return weights_;
        }
        const WeightsMap & weights() const{
            return weights_;
        }

        const Graph & graph() const{
            return graph_;
        }

        LiftedGraph & liftedGraph(){
            return graph_;
        }
        const LiftedGraph & liftedGraph() const{
            return graph_;
        }
        

    private:
        const Graph & graph_;
        LiftedGraph liftedGraph_;
        WeightsMap weights_;
        ToContiguousNodes toContiguousNodes_;



    };

} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
