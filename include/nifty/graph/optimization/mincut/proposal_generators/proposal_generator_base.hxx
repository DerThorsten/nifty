
#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_BASE_HXX2
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_BASE_HXX2


#include <memory>
#include "nifty/graph/optimization/mincut/lifted_multicut_base.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{



    template<class OBJECTIVE>
    class ProposalGeneratorBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef MincutBase<ObjectiveType> MincutBaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        typedef typename MincutBaseType::NodeLabels NodeLabels;
    
        virtual ~ProposalGeneratorBase(){}

        virtual void generateProposal( const NodeLabels & currentBest,NodeLabels & labels, const size_t tid) = 0;

    private:
    }; 

    
    
    

}
}
}
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_BASE_HXX2