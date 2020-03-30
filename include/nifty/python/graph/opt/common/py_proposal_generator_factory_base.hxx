#pragma once


#include "pybind11/pybind11.h"

#include "nifty/graph/opt/common/proposal_generators/proposal_generator_factory_base.hxx"
#include "nifty/graph/opt/common/proposal_generators/proposal_generator_factory.hxx"

namespace nifty {
namespace graph {
namespace opt{
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
    std::shared_ptr<ProposalGeneratorBaseType> createShared(const ObjectiveType & objective, const std::size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<ProposalGeneratorBaseType>,     /* Return type */
            ProposalGeneratorFactoryBase<ObjectiveType>,        /* Parent class */
            createShared,                                /* Name of function */
            objective, numberOfThreads                      /* Argument(s) */
        );
    }
    ProposalGeneratorBaseType * create(const ObjectiveType & objective, const std::size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            ProposalGeneratorBaseType* ,                    /* Return type */
            ProposalGeneratorFactoryBase<ObjectiveType>,        /* Parent class */
            create,                                   /* Name of function */
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
py::class_<typename PROPOSAL_GENERATOR::SettingsType>
exportCCProposalGenerator(
    py::module & module,
    const std::string & clsName,
    const std::string & objName
){
    typedef PROPOSAL_GENERATOR ProposalGeneratorType;
    typedef typename ProposalGeneratorType::ObjectiveType   ObjectiveType;
    typedef typename ProposalGeneratorType::SettingsType        SettingsType;
    typedef ProposalGeneratorFactory<ProposalGeneratorType> Factory;


    const std::string settingsName = std::string("__") + clsName + std::string("SettingsType") + objName;
    const std::string factoryBaseName = std::string("__ProposalGeneratorFactoryBase")+objName;
    const std::string factoryName = clsName + std::string("Factory") + objName;


     // settings
    auto settingsCls = py::class_< SettingsType >(module, settingsName.c_str())
    ;

    // factory
    py::object factoryBase = module.attr(factoryBaseName.c_str());
    py::class_<Factory, std::shared_ptr<Factory> >(module, factoryName.c_str(),  factoryBase)
        .def(py::init<const SettingsType &>(),
            py::arg_t<SettingsType>("setttings",SettingsType())
        )
    ;


    return settingsCls;

}





} // namespace common
} // namespace opt
} // namespace graph
} // namespace nifty


