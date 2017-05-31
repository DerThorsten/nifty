
#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_FACTORY_BASE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_FACTORY_BASE_HXX


#include <memory>


#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{


    template<class OBJECTIVE>
    class ProposalGeneratorFactoryBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;

        virtual ~ProposalGeneratorFactoryBase(){}
        virtual std::shared_ptr<ProposalGeneratorBaseType> createSharedPtr(const ObjectiveType & objective, const size_t numberOfThreads) = 0;
        virtual ProposalGeneratorBaseType *                createRawPtr(   const ObjectiveType & objective, const size_t numberOfThreads) = 0;
    };



}
}
}
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_FACTORY_BASE_HXX