
#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_FUSION_MOVE_BASED_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_FUSION_MOVE_BASED_HXX


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


//#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_base.hxx"
//#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_factory_base.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{









    template<class OBJECTIVE>
    class FusionMoveBased : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> BaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        
        typedef typename BaseType::VisitorBase VisitorBase;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::NodeLabels NodeLabels;

        //typedef ProposalGeneratorFactoryBase<ObjectiveType> ProposalGeneratorFactoryBaseType;

    private:


    

    public:

        struct Settings{
            //std::shared_ptr<ProposalGeneratorFactoryBaseType> proposalGeneratorFactory;
        };



        virtual ~FusionMoveBased(){}
        FusionMoveBased(const ObjectiveType & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const ObjectiveType & objective() const;





 


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("FusionMoveBased");
        }


    private:




        const ObjectiveType & objective_;
        Settings settings_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        NodeLabels * currentBest_;
    };

    
    template<class OBJECTIVE>
    FusionMoveBased<OBJECTIVE>::
    FusionMoveBased(
        const ObjectiveType & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        liftedGraph_(objective.liftedGraph()),
        currentBest_(nullptr)
    {


    }

    template<class OBJECTIVE>
    void FusionMoveBased<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        
        currentBest_ = &nodeLabels;

        VisitorProxy visitorProxy(visitor);
        visitorProxy.begin(this);


        visitorProxy.end(this);     
    }

    template<class OBJECTIVE>
    const typename FusionMoveBased<OBJECTIVE>::ObjectiveType &
    FusionMoveBased<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 


    
} // lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_FUSION_MOVE_BASED_HXX
