
#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_WATERSHED_PROPOSAL_GENERATOR_BASE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_WATERSHED_PROPOSAL_GENERATOR_BASE_HXX


#include <memory>
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{



    template<class OBJECTIVE>
    class WatershedProposalGenerator : public ProposalGeneratorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> LiftedMulticutBaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        typedef typename LiftedMulticutBaseType::NodeLabels NodeLabels;
    
        struct Settings{

        };

        WatershedProposalGenerator(
            const ObjectiveType & objective, 
            const size_t numberOfThreads,
            const Settings & settings  = Settings()
        ){

        }

        virtual ~WatershedProposalGenerator(){}
        virtual void generateProposal( const NodeLabels & currentBest,NodeLabels & labels, const size_t tid){

        }
    private:

    }; 





}
}
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_WATERSHED_PROPOSAL_GENERATOR_BASE_HXX