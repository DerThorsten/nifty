#pragma once

#include "nifty/graph/opt/lifted_multicut/proposal_generators/proposal_generator_factory_base.hxx"


namespace nifty {
namespace graph {
namespace opt{
namespace lifted_multicut{







template<class OBJECTIVE>
class PyProposalGeneratorFactoryBase : public ProposalGeneratorFactoryBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<ObjectiveType>::LiftedMulticutFactory;
    typedef OBJECTIVE ObjectiveType;
    typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;
    /* Trampoline (need one for each virtual function) */
    std::shared_ptr<ProposalGeneratorBaseType> createShared(const ObjectiveType & objective, const std::size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<ProposalGeneratorBaseType>,     /* Return type */
            ProposalGeneratorFactoryBase<ObjectiveType>,        /* Parent class */
            createShared,                                              /* Name of function */
            objective, numberOfThreads                                           /* Argument(s) */
        );
    }
    ProposalGeneratorBaseType * create(const ObjectiveType & objective, const std::size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            ProposalGeneratorBaseType* ,                    /* Return type */
            ProposalGeneratorFactoryBase<ObjectiveType>,        /* Parent class */
            create,                                                 /* Name of function */
            objective, numberOfThreads                                           /* Argument(s) */
        );
    }
};


} // namespace lifted_mutlicut
} // namespace opt
} // namespace graph
} // namespace nifty

