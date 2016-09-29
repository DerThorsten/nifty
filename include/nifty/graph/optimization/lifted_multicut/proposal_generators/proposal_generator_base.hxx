
#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_BASE_HXX2
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_BASE_HXX2


#include <memory>
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{



    template<class OBJECTIVE>
    class ProposalGeneratorBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> LiftedMulticutBaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        typedef typename LiftedMulticutBaseType::NodeLabels NodeLabels;
    private:
        virtual ~ProposalGeneratorBase(){}
        virtual void generateProposal( const NodeLabels & currentBest,NodeLabels & labels) = 0;
    };

}
}
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_BASE_HXX2