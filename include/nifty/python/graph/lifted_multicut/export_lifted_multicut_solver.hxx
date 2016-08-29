#pragma once
#ifndef NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_EXPORT_LIFTED_MULTICUT_SOLVER_HXX
#define NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_EXPORT_LIFTED_MULTICUT_SOLVER_HXX



#include <pybind11/pybind11.h>

#include "nifty/python/graph/lifted_multicut/lifted_multicut_objective.hxx"
#include "py_lifted_multicut_factory.hxx"
#include "py_lifted_multicut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class SOLVER>
    py::class_<typename SOLVER::Settings>  exportLiftedMulticutSolver(
        py::module & liftedMulticutModule,
        const std::string & solverName
    ){

        typedef SOLVER Solver;
        typedef typename Solver::Objective ObjectiveType;
        typedef typename Solver::Settings Settings;
        typedef LiftedMulticutFactory<Solver> Factory;
        typedef LiftedMulticutFactoryBase<ObjectiveType> LmcFactoryBase;

        const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("LiftedMulticutFactoryBase")+objName;
        const std::string solverBaseName = std::string("LiftedMulticutBase") + objName;
        
        const std::string sName = solverName + objName;
        const std::string settingsName = solverName + std::string("Settings") + objName;
        const std::string factoryName = solverName + std::string("Factory") + objName;
        std::string factoryFactoryName = factoryName;
        factoryFactoryName[0] = std::tolower(factoryFactoryName[0]);


        py::object factoryBase = liftedMulticutModule.attr(factoryBaseName.c_str());
        py::object solverBase = liftedMulticutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< Settings >(liftedMulticutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(liftedMulticutModule, factoryName.c_str(),  factoryBase)
            .def(py::init<const Settings &>(),
                py::arg_t<Settings>("setttings",Settings())
            )
        ;

        // solver
        py::class_<Solver >(liftedMulticutModule, sName.c_str(),  solverBase)
            //.def(py::init<>())
        ;

        return settingsCls;

    }

}
}
}



#endif /* NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_EXPORT_LIFTED_MULTICUT_SOLVER_HXX */
