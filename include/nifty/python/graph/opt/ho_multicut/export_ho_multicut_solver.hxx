#pragma once


#include <pybind11/pybind11.h>

#include "nifty/python/graph/opt/solver_docstring.hxx"
#include "nifty/python/graph/opt/ho_multicut/ho_multicut_objective.hxx"
//#include "nifty/python/graph/opt/common/py_solver_factory_base.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"
#include "py_ho_multicut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{

    template<class SOLVER>
    py::class_<typename SOLVER::SettingsType>  exportHoMulticutSolver(
        py::module & hoHoMulticutModule,
        const std::string & solverName,
        nifty::graph::opt::SolverDocstringHelper docHelper = nifty::graph::opt::SolverDocstringHelper()
    ){

        typedef SOLVER Solver;
        typedef typename Solver::ObjectiveType ObjectiveType;
        typedef typename Solver::SettingsType SettingsType;
        typedef nifty::graph::opt::common::SolverFactory<Solver> Factory;


        const auto objName = HoMulticutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("SolverFactoryBase")+objName;
        const std::string solverBaseName = std::string("HoMulticutBase") + objName;
        
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




        py::object factoryBase = hoHoMulticutModule.attr(factoryBaseName.c_str());
        py::object solverBase = hoHoMulticutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< SettingsType >(hoHoMulticutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(
            hoHoMulticutModule,
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
            hoHoMulticutModule, sName.c_str(),  
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
