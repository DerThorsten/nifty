#pragma once

#include <memory>
#include "nifty/graph/opt/common/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace common{




    template<class OBJECTIVE>
    class ProposalGeneratorFactoryBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;

        virtual ~ProposalGeneratorFactoryBase(){}
        virtual std::shared_ptr<ProposalGeneratorBaseType> createShared(const ObjectiveType & objective, const size_t numberOfThreads) = 0;
        virtual ProposalGeneratorBaseType *                create(   const ObjectiveType & objective, const size_t numberOfThreads) = 0;
    };



} // namespace nifty::graph::opt::common
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

