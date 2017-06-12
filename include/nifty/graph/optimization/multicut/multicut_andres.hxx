#pragma once

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/ufd/ufd.hxx"

// andres::graph includes
#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"
#include "andres/graph/multicut/greedy-additive.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{
    
    template<class OBJECTIVE>
    class MulticutAndres : public MulticutBase<OBJECTIVE>
    {
    public: 
        typedef OBJECTIVE Objective;
        typedef MulticutBase<OBJECTIVE> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef andres::graph::Graph<> Graph;

        MulticutAndres(const Objective & objective);

        virtual void optimize(NodeLabelsType & nodelabels, VisitorBaseType * visitor){}

        virtual const Objective & objective() const {return objective_;}
        virtual const NodeLabelsType & currentBestNodeLabels() {return *currentBest_;}
        virtual std::string name() const {return "MulticutAndres";}
        
    protected:
        void nodeLabelsToEdgeLabels(std::vector<char> & edgeLabels);
        void edgeLabelsToNodeLabels(const std::vector<char> & edgeLabels);
        NodeLabelsType * currentBest_;
        Graph graph_;

    private:
        void initGraph();
        const Objective & objective_;
        ufd::Ufd<uint64_t> ufd_;
    };


    template<class OBJECTIVE>
    MulticutAndres<OBJECTIVE>::MulticutAndres(
        const Objective & objective
    ) : currentBest_(nullptr),
        graph_(objective.graph().numberOfNodes()),
        objective_(objective),
        ufd_(graph_.numberOfVertices())
    {
        initGraph();
    }


    template<class OBJECTIVE>
    void MulticutAndres<OBJECTIVE>::initGraph(){
        const auto & objGraph = objective_.graph();
        for(auto e : objGraph.edges()){
            const auto & uv = objGraph.uv(e);
            graph_.insertEdge(uv.first, uv.second);
        }
    }


    template<class OBJECTIVE>
    void MulticutAndres<OBJECTIVE>::nodeLabelsToEdgeLabels(std::vector<char> & edgeLabels) {
        const auto & nodeLabels = *currentBest_;
        for(auto edgeId = 0; edgeId <  graph_.numberOfEdges(); ++edgeId) {
            const auto u = graph_.vertexOfEdge(edgeId,0); 
            const auto v = graph_.vertexOfEdge(edgeId,1);
            edgeLabels[edgeId] = nodeLabels[u] != nodeLabels[v];
        }
    }


    template<class OBJECTIVE>
    void MulticutAndres<OBJECTIVE>::edgeLabelsToNodeLabels(const std::vector<char> & edgeLabels) {
        for(auto edgeId = 0; edgeId <  graph_.numberOfEdges(); ++edgeId) {
            if(!edgeLabels[edgeId]){
                ufd_.merge( graph_.vertexOfEdge(edgeId,0), graph_.vertexOfEdge(edgeId,1) );
            }
        }
        ufd_.elementLabeling(currentBest_->begin());
    }



    template<class OBJECTIVE>
    class MulticutAndresGreedyAdditive : public MulticutAndres<OBJECTIVE>
    {
    public:
        
        typedef OBJECTIVE Objective;
        typedef MulticutAndres<OBJECTIVE> Base;
        typedef typename Base::NodeLabelsType NodeLabelsType;
        typedef typename Base::VisitorBaseType VisitorBaseType;
        
        struct SettingsType {};
        MulticutAndresGreedyAdditive(const Objective & objective, const SettingsType & settings = SettingsType());
        
        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const Objective & objective() const {return Base::objective();}
        virtual const NodeLabelsType & currentBestNodeLabels() {return Base::currentBestNodeLabels();}
        virtual std::string name() const {return "MulticutAndresGreedyAdditive";}

    };

    template<class OBJECTIVE>
    MulticutAndresGreedyAdditive<OBJECTIVE>::
    MulticutAndresGreedyAdditive(const Objective & objective, const SettingsType &)
    : Base(objective)
    {}

    template<class OBJECTIVE>
    void MulticutAndresGreedyAdditive<OBJECTIVE>::optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){  
        //VisitorProxyType visitorProxy(visitor);
        Base::currentBest_ = &nodeLabels;
        
        if(Base::graph_.numberOfEdges()>0){
            std::vector<char> edgeLabels( Base::graph_.numberOfEdges() );
            Base::nodeLabelsToEdgeLabels(edgeLabels);
            andres::graph::multicut::greedyAdditiveEdgeContraction(Base::graph_, objective().weights(), edgeLabels);
            Base::edgeLabelsToNodeLabels(edgeLabels);
        }
    }
    
    
    template<class OBJECTIVE>
    class MulticutAndresKernighanLin : public MulticutAndres<OBJECTIVE>
    {
    public:
        
        typedef OBJECTIVE Objective;
        typedef MulticutAndres<OBJECTIVE> Base;
        typedef typename Base::NodeLabelsType NodeLabelsType;
        typedef typename Base::VisitorBaseType VisitorBaseType;

        typedef andres::graph::multicut::KernighanLinSettings KlSettings;

        struct SettingsType {
            size_t numberOfInnerIterations { std::numeric_limits<size_t>::max() };
            size_t numberOfOuterIterations { 100 };
            double epsilon { 1e-6 };
            bool verbose { false };
            bool greedyWarmstart{true};
        };

        MulticutAndresKernighanLin(const Objective & objective, const SettingsType & settings = SettingsType());
        
        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const Objective & objective() const {return Base::objective();}
        virtual const NodeLabelsType & currentBestNodeLabels() {return Base::currentBestNodeLabels();}
        virtual std::string name() const {return "MulticutAndresKernighanLin";}

    private:
        SettingsType settings_;
        KlSettings klSettings_;
    };

    template<class OBJECTIVE>
    MulticutAndresKernighanLin<OBJECTIVE>::
    MulticutAndresKernighanLin(const Objective & objective, const SettingsType & settings)
    : Base(objective),
      settings_(settings),
      klSettings_()
    {
        klSettings_.numberOfInnerIterations = settings_.numberOfInnerIterations;
        klSettings_.numberOfOuterIterations = settings_.numberOfOuterIterations;
        klSettings_.epsilon = settings_.epsilon;
        klSettings_.verbose = settings_.verbose;
    }

    template<class OBJECTIVE>
    void MulticutAndresKernighanLin<OBJECTIVE>::optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){  
        //VisitorProxyType visitorProxy(visitor);
        Base::currentBest_ = &nodeLabels;
        
        if(Base::graph_.numberOfEdges()>0){
            std::vector<char> edgeLabels( Base::graph_.numberOfEdges() );
            Base::nodeLabelsToEdgeLabels(edgeLabels);
            if(settings_.greedyWarmstart){
                andres::graph::multicut::greedyAdditiveEdgeContraction(Base::graph_, objective().weights(), edgeLabels);
            }
            andres::graph::multicut::kernighanLin(Base::graph_, objective().weights(), edgeLabels, edgeLabels, klSettings_);
            Base::edgeLabelsToNodeLabels(edgeLabels);
        }
    }
} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
