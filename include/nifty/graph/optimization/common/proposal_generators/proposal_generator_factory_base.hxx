#pragma once

#include <memory>
#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace common{




    template<class OBJECTIVE>
    class ProposalGeneratorFactoryBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;

        virtual ~ProposalGeneratorFactoryBase(){}
        virtual std::shared_ptr<ProposalGeneratorBaseType> createSharedPtr(const ObjectiveType & objective, const size_t numberOfThreads) = 0;
        virtual ProposalGeneratorBaseType *                createRawPtr(   const ObjectiveType & objective, const size_t numberOfThreads) = 0;
    };



} // namespace nifty::graph::optimization::common
} // namespacen ifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

