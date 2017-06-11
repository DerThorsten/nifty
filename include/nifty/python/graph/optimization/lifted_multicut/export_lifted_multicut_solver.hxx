#pragma once



#include <pybind11/pybind11.h>

#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/optimization/common/solver_factory.hxx"
#include "py_lifted_multicut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{

    template<class SOLVER>
    py::class_<typename SOLVER::SettingsType>  exportLiftedMulticutSolver(
        py::module & liftedMulticutModule,
        const std::string & solverName
    ){

        typedef SOLVER Solver;
        typedef typename Solver::ObjectiveType ObjectiveType;
        typedef typename Solver::SettingsType SettingsType;
        typedef nifty::graph::optimization::common::SolverFactory<Solver> Factory;


        const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("SolverFactoryBase")+objName;
        const std::string solverBaseName = std::string("LiftedMulticutBase") + objName;
        
        const std::string sName = solverName + objName;
        const std::string settingsName = solverName + std::string("SettingsType") + objName;
        const std::string factoryName = solverName + std::string("Factory") + objName;
        std::string factoryFactoryName = factoryName;
        factoryFactoryName[0] = std::tolower(factoryFactoryName[0]);


        py::object factoryBase = liftedMulticutModule.attr(factoryBaseName.c_str());
        py::object solverBase = liftedMulticutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< SettingsType >(liftedMulticutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(liftedMulticutModule, factoryName.c_str(),  factoryBase)
            .def(py::init<const SettingsType &>(),
                py::arg_t<SettingsType>("setttings",SettingsType())
            )
        ;

        // solver
        py::class_<Solver >(liftedMulticutModule, sName.c_str(),  solverBase)
            //.def(py::init<>())
        ;

        return settingsCls;

    }

}
} // namespace optimization
}
}



