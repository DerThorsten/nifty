#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/opt/minstcut/minstcut_base.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"


#include "graph.h"


namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{

    template<class OBJECTIVE>
    class MinstcutMaxflow : public MinstcutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef MinstcutBase<OBJECTIVE> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef typename ObjectiveType::GraphType GraphType;
    
    private:

        typedef float captype;
        typedef float tcaptype;
        typedef float flowtype;

        typedef detail_graph::NodeIndicesToContiguousNodeIndices<GraphType> DenseIds;

    public:

        MinstcutMaxflow(ObjectiveType & objective);
        ~MinstcutMaxflow(){}

    private:

        void initializeMaxflow();

        const ObjectiveType & objective_;
        const GraphType & graph_;

        Graph<captype, tcaptype, flowtype> maxflow_;

    };


    template<class OBJECTIVE>
    MinstcutMaxflow<OBJECTIVE>::
    MinstcutMaxflow(
        const ObjectiveType & objective,
    )
    :   objective_(objective),
        graph_(objective.graph()),
        maxflow_(graph_.numberOfNodes(), graph_.numberOfEdges())

    {
        const auto & unaries = objective_.unaries();
        const auto & weights = objective_.weights();

        // initialize maxflow
        maxflow_.add_node(graph_.numberOfNodes());


        for(auto node : graph_.nodes()){
            const auto & uVec = unaries[node];
            const auto e0 = uVec.first;
            const auto e1 = uVec.second;

            if(e0 < e1){
                const auto c0 = 0.0;
                const auto c1 = e1 - e0;
                maxflow_.add_tweight(node, c0, c1);
            }
            else{
                const auto c1 = 0.0;
                const auto c0 = e0 - e1;
                maxflow_.add_tweights(node, c0, c1)
            }
        }


        for(auto edge : graph_.edges()){
            const auto uv = graph_.uvIds(edge);
            const auto w = weights[edge];
            maxflow_.add_edge(uv.first, uv.second, w, 0.0);
        }
    }

    template<class OBJECTIVE>
    void MinstcutMaxflow<OBJECTIVE>::
    optimize(
        NodeLabelsType & nodeLabels, VisitorBaseType * visitor
        ){

        VisitorProxyType visitorProxy(visitor);
        visitorProxy.begin(this);

        graph_.maxflow();

        for (auto node : graph_nodes()) {
            if (graph_.what_segment(node) == maxflow_::SOURCE) {
                nodeLabels[node] = 0.;
            }
            else {
                nodeLabels[node] = 1.;
            }
        }
    }


