#pragma once

// wrapper around the andres kernighan lin multicut solver

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"
#include "andres/graph/graph.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty{
namespace graph{
    
    template<class OBJECTIVE>
    class MulticutKernighanLin : public MulticutBase<OBJECTIVE>
    {
    public: 
        typedef OBJECTIVE Objective;
        typedef MulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;
        typedef andres::graph::Graph<> Graph;
        
        struct Settings{
            bool verbose{0};
            size_t numberOfInnerIterations{std::numeric_limits<size_t>::max()};
            size_t numberOfOuterIterations{100};
            double epsilon{1e-7};
        };

        MulticutKernighanLin(const Objective & objective, const Settings & settings = Settings());

        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);

        virtual const Objective & objective() const {return objective_;}
        virtual const NodeLabels & currentBestNodeLabels() {return *currentBest_;}
        virtual std::string name() const {return "MulticutKernighanLin";}

    private:
        void initGraph();

        const Objective & objective_;
        Graph graph_;
        Settings settings_;
        NodeLabels * currentBest_;
        ufd::Ufd<uint64_t> ufd_;
    };


    template<class OBJECTIVE>
    MulticutKernighanLin<OBJECTIVE>::MulticutKernighanLin(
        const Objective & objective, 
        const Settings & settings
    ) : objective_(objective),
        graph_(objective.graph().numberOfNodes()),
        settings_(settings),
        ufd_(graph_.numberOfVertices())
    {
        initGraph();
    }


    template<class OBJECTIVE>
    void MulticutKernighanLin<OBJECTIVE>::initGraph(){
        const auto & objGraph = objective_.graph();
        for(auto e : objGraph.edges()){
            const auto & uv = objGraph.uv(e);
            graph_.insertEdge(uv.first, uv.second);
        }
    }


    template<class OBJECTIVE>
    void MulticutKernighanLin<OBJECTIVE>::optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  
        VisitorProxy visitorProxy(visitor);
        currentBest_ = &nodeLabels;
        
        // TODO for now the visitor is doing nothing, but we should implement one, that is
        // compatible with lp_mp visitor
        //visitorProxy.begin(this);
        
        if(graph_.numberOfEdges()>0){
        
            // get the edge labels from thei initial node labels
            std::vector<char> edgeLabels( graph_.numberOfEdges() );
            for(auto edgeId = 0; edgeId <  graph_.numberOfEdges(); ++edgeId) {
                const auto u = graph_.vertexOfEdge(edgeId,0); 
                const auto v = graph_.vertexOfEdge(edgeId,1);
                edgeLabels[edgeId] = nodeLabels[u] != nodeLabels[v];
            }
            
            andres::graph::multicut::kernighanLin(graph_, objective_.weights(), edgeLabels, edgeLabels);
            
            // resulting edge labels to node labels via ufd
            for(auto edgeId = 0; edgeId <  graph_.numberOfEdges(); ++edgeId) {
                if(!edgeLabels[edgeId]){
                    ufd_.merge( graph_.vertexOfEdge(edgeId,0), graph_.vertexOfEdge(edgeId,1) );
                }
            }
        
        }
        ufd_.elementLabeling(currentBest_->begin());
    }

}
}
