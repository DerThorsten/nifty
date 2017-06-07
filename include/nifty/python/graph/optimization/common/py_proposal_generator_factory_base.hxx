#pragma once


#include "pybind11/pybind11.h"

#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_factory_base.hxx"
#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_factory.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace common{






template<class OBJECTIVE>
class PyProposalGeneratorFactoryBase : public ProposalGeneratorFactoryBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<Objective>::LiftedMulticutFactory;
    typedef OBJECTIVE ObjectiveType;
    typedef typename ObjectiveType::GraphType GraphType;
    typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;
    /* Trampoline (need one for each virtual function) */
    std::shared_ptr<ProposalGeneratorBaseType> createSharedPtr(const ObjectiveType & objective, const size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<ProposalGeneratorBaseType>,     /* Return type */
            ProposalGeneratorFactoryBase<ObjectiveType>,        /* Parent class */
            createSharedPtr,                                /* Name of function */
            objective, numberOfThreads                      /* Argument(s) */
        );
    }
    ProposalGeneratorBaseType * createRawPtr(const ObjectiveType & objective, const size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            ProposalGeneratorBaseType* ,                    /* Return type */
            ProposalGeneratorFactoryBase<ObjectiveType>,        /* Parent class */
            createRawPtr,                                   /* Name of function */
            objective, numberOfThreads                     /* Argument(s) */
        );
    }
};



template<class OBJECTIVE>
void exportCCProposalGeneratorFactoryBaseT(
    pybind11::module & module,
    const std::string & objName
) {
    typedef OBJECTIVE ObjectiveType;
    //const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();
    const auto clsName = std::string("__ProposalGeneratorFactoryBase") + objName;

    typedef ProposalGeneratorFactoryBase<ObjectiveType> PropGenFactoryBase;
    typedef PyProposalGeneratorFactoryBase<ObjectiveType> PyPropGenFactoryBase;
    
    // base factory
    pybind11::class_<
        PropGenFactoryBase, 
        std::shared_ptr<PropGenFactoryBase>, 
       PyPropGenFactoryBase
    >  proposalsGenFactoryBase(module, clsName.c_str());
    
    proposalsGenFactoryBase
        .def(pybind11::init<>())
    ;
}



template<class PROPOSAL_GENERATOR>
py::class_<typename PROPOSAL_GENERATOR::Settings> 
exportCCProposalGenerator(
    py::module & module,
    const std::string & clsName,
    const std::string & objName
){
    typedef PROPOSAL_GENERATOR ProposalGeneratorType;
    typedef typename ProposalGeneratorType::ObjectiveType   ObjectiveType;
    typedef typename ProposalGeneratorType::Settings        Settings;
    typedef ProposalGeneratorFactory<ProposalGeneratorType> Factory;


    const std::string settingsName = std::string("__") + clsName + std::string("Settings") + objName;
    const std::string factoryBaseName = std::string("__ProposalGeneratorFactoryBase")+objName;
    const std::string factoryName = clsName + std::string("Factory") + objName;


     // settings
    auto settingsCls = py::class_< Settings >(module, settingsName.c_str())
    ;

    // factory
    py::object factoryBase = module.attr(factoryBaseName.c_str());
    py::class_<Factory, std::shared_ptr<Factory> >(module, factoryName.c_str(),  factoryBase)
        .def(py::init<const Settings &>(),
            py::arg_t<Settings>("setttings",Settings())
        )
    ;


    return settingsCls;

}   





} // namespace common
} // namespace optimization
} // namespace graph
} // namespace nifty


