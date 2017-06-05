
#pragma once


#include <memory>


#include "nifty/graph/optimization/mincut/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{


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

