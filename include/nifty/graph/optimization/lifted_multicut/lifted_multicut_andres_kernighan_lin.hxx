// reimplementation of kerninhanlin in 
// https://github.com/bjoern-andres/graph

#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_ANDRES_KERNIGHAN_LIN_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_ANDRES_KERNIGHAN_LIN_HXX


#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <stack>

#include "vigra/priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/components.hxx"


#include "andres/graph/graph.hxx"
#include "andres/graph/multicut-lifted/kernighan-lin.hxx"



namespace nifty{
namespace graph{
namespace lifted_multicut{










    template<class OBJECTIVE>
    class LiftedMulticutAndresKernighanLin : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> BaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        
        typedef typename BaseType::VisitorBase VisitorBase;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::NodeLabels NodeLabels;

    private:


    

    public:

        typedef andres::graph::multicut_lifted::KernighanLinSettings Settings;



        virtual ~LiftedMulticutAndresKernighanLin(){}
        LiftedMulticutAndresKernighanLin(const ObjectiveType & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const ObjectiveType & objective() const;





 


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("LiftedMulticutAndresKernighanLin");
        }


    private:




        const ObjectiveType & objective_;
        Settings settings_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        NodeLabels * currentBest_;

        andres::graph::Graph<> aGraph_;
        andres::graph::Graph<> aLiftedGraph_;
        std::vector<double> edgeCosts_;
    };

    
    template<class OBJECTIVE>
    LiftedMulticutAndresKernighanLin<OBJECTIVE>::
    LiftedMulticutAndresKernighanLin(
        const ObjectiveType & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        liftedGraph_(objective.liftedGraph()),
        currentBest_(nullptr),
        aGraph_(objective.graph().nodeIdUpperBound()+1),
        aLiftedGraph_(objective.graph().nodeIdUpperBound()+1),
        edgeCosts_(objective.liftedGraph().numberOfEdges(),0)
    {
        for(const auto edge : graph_.edges()){
            const auto uv = graph_.uv(edge);
            aGraph_.insertEdge(uv.first, uv.second);
        }

        auto c = 0;
        for(const auto edge : liftedGraph_.edges()){
            const auto uv = liftedGraph_.uv(edge);
            aLiftedGraph_.insertEdge(uv.first, uv.second);
            edgeCosts_[c] = objective_.weights()[edge];
            ++c;
        }
    }

    template<class OBJECTIVE>
    void LiftedMulticutAndresKernighanLin<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        
        currentBest_ = &nodeLabels;

        VisitorProxy visitorProxy(visitor);
        //visitorProxy.addLogNames({"violatedCycleConstraints","violatedCutConstraints"});
        visitorProxy.begin(this);

        typedef std::vector<uint8_t> ELA;


        
        std::vector<uint8_t> ioLabels(aLiftedGraph_.numberOfEdges());
        
        std::vector<std::size_t> edegInALiftedGraph(aGraph_.numberOfEdges());

        

        for (std::size_t i = 0; i < aGraph_.numberOfEdges(); ++i){
            const auto v0 = aGraph_.vertexOfEdge(i, 0);
            const auto v1 = aGraph_.vertexOfEdge(i, 1);
            edegInALiftedGraph[i] = aLiftedGraph_.findEdge(v0, v1).second;
        }


        for (std::size_t i = 0; i < aLiftedGraph_.numberOfEdges(); ++i){
            const auto v0 = aLiftedGraph_.vertexOfEdge(i, 0);
            const auto v1 = aLiftedGraph_.vertexOfEdge(i, 1);
            ioLabels[i] = nodeLabels[v0] != nodeLabels[v1] ? 1 : 0;
        }

        settings_.verbose = false;

        
        andres::graph::multicut_lifted::kernighanLin(
            aGraph_,
            aLiftedGraph_,
            edgeCosts_,
            ioLabels,
            ioLabels,
            settings_
        );
        
       

        struct SubgraphWithCut { // a subgraph with cut mask
            SubgraphWithCut(const ELA& labels, std::vector<std::size_t> const& edegInALiftedGraph)
                : labels_(labels), edegInALiftedGraph_(edegInALiftedGraph)
                {}
            bool vertex(const std::size_t v) const
                { return true; }
            bool edge(const std::size_t e) const
                { return labels_[edegInALiftedGraph_[e]] == 0; }

            std::vector<std::size_t> const& edegInALiftedGraph_;
            const ELA& labels_;
        };

        // build decomposition based on the current multicut
        andres::graph::ComponentsByPartition<andres::graph::Graph<> > components;
        components.build(aGraph_, SubgraphWithCut(ioLabels, edegInALiftedGraph));

        for(const auto node : graph_.nodes()){
            nodeLabels[node] = components.partition_.find(node);
        }


        visitorProxy.end(this);     
    }

    template<class OBJECTIVE>
    const typename LiftedMulticutAndresKernighanLin<OBJECTIVE>::ObjectiveType &
    LiftedMulticutAndresKernighanLin<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 


    
} // lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_ANDRES_KERNIGHAN_LIN_HXX
