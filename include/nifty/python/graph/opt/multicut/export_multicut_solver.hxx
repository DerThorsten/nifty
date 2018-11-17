#pragma once


#include <pybind11/pybind11.h>

#include "nifty/python/graph/opt/solver_docstring.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
//#include "nifty/python/graph/opt/common/py_solver_factory_base.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"
#include "py_multicut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    template<class SOLVER>
    py::class_<typename SOLVER::SettingsType>  exportMulticutSolver(
        py::module & multicutModule,
        const std::string & solverName,
        nifty::graph::opt::SolverDocstringHelper docHelper = nifty::graph::opt::SolverDocstringHelper()
    ){

        typedef SOLVER Solver;
        typedef typename Solver::ObjectiveType ObjectiveType;
        typedef typename Solver::SettingsType SettingsType;
        typedef nifty::graph::opt::common::SolverFactory<Solver> Factory;

        const auto objName = MulticutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("SolverFactoryBase")+objName;
        const std::string solverBaseName = std::string("MulticutBase") + objName;

        const std::string sName = solverName + objName;
        const std::string settingsName = std::string("__") + solverName + std::string("SettingsType") + objName;
        const std::string factoryName = solverName + std::string("Factory") + objName;
        std::string factoryFactoryName = factoryName;
        factoryFactoryName[0] = std::tolower(factoryFactoryName[0]);

        // setup dochelper
        docHelper.factoryBaseClsName = factoryBaseName;
        docHelper.solverBaseClsName = solverBaseName;
        docHelper.solverClsName = sName;
        docHelper.factoryClsName = factoryName;
        docHelper.factoryBaseClsName = factoryBaseName;
        docHelper.factoryBaseClsName = factoryBaseName;
        docHelper.factoryFactoryName = factoryFactoryName;

        py::object factoryBase = multicutModule.attr(factoryBaseName.c_str());
        py::object solverBase = multicutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< SettingsType >(multicutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(
            multicutModule,
            factoryName.c_str(),
            factoryBase,
            docHelper.factoryDocstring<Factory>().c_str()
        )
            .def(py::init<const SettingsType &>(),
                py::arg_t<SettingsType>("setttings",SettingsType())
            )
        ;

        // solver
        py::class_<Solver >(
            multicutModule, sName.c_str(),
            solverBase,
            docHelper.solverDocstring<Solver>().c_str()
        )
            //.def(py::init<>())
        ;

        return settingsCls;
    }

} // namespace multicut
} // namespace opt
}
}
