#pragma once


#include <pybind11/pybind11.h>

#include "nifty/python/graph/optimization/solver_docstring.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "py_multicut_factory.hxx"
#include "py_multicut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

    template<class SOLVER>
    py::class_<typename SOLVER::Settings>  exportMulticutSolver(
        py::module & multicutModule,
        const std::string & solverName,
        nifty::graph::optimization::SolverDocstringHelper docHelper = nifty::graph::optimization::SolverDocstringHelper()
    ){

        typedef SOLVER Solver;
        typedef typename Solver::Objective ObjectiveType;
        typedef typename Solver::Settings Settings;
        typedef MulticutFactory<Solver> Factory;
        typedef MulticutFactoryBase<ObjectiveType> McFactoryBase;

        const auto objName = MulticutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("MulticutFactoryBase")+objName;
        const std::string solverBaseName = std::string("MulticutBase") + objName;
        
        const std::string sName = solverName + objName;
        const std::string settingsName = std::string("__") + solverName + std::string("Settings") + objName;
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
        auto settingsCls = py::class_< Settings >(multicutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(
            multicutModule,
            factoryName.c_str(),  
            factoryBase,
            docHelper.factoryDocstring<Factory>().c_str()
        )
            .def(py::init<const Settings &>(),
                py::arg_t<Settings>("setttings",Settings())
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
} // namespace optimization
}
}
