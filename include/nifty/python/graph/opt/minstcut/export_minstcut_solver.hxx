#pragma once



#include <pybind11/pybind11.h>

#include "nifty/python/graph/opt/minstcut/minstcut_objective.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"
#include "py_minstcut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{

    template<class SOLVER>
    py::class_<typename SOLVER::SettingsType>  exportMinstcutSolver(
        py::module & minstcutModule,
        const std::string & solverName
    ){

        typedef SOLVER Solver;
        typedef typename Solver::ObjectiveType ObjectiveType;
        typedef typename Solver::SettingsType SettingsType;
        typedef nifty::graph::opt::common::SolverFactory<Solver> Factory;


        const auto objName = MinstcutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("SolverFactoryBase")+objName;
        const std::string solverBaseName = std::string("MinstcutBase") + objName;
        
        const std::string sName = solverName + objName;
        const std::string settingsName = solverName + std::string("SettingsType") + objName;
        const std::string factoryName = solverName + std::string("Factory") + objName;
        std::string factoryFactoryName = factoryName;
        factoryFactoryName[0] = std::tolower(factoryFactoryName[0]);


        py::object factoryBase = minstcutModule.attr(factoryBaseName.c_str());
        py::object solverBase = minstcutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< SettingsType >(minstcutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(minstcutModule, factoryName.c_str(),  factoryBase)
            .def(py::init<const SettingsType &>(),
                py::arg_t<SettingsType>("setttings",SettingsType())
            )
        ;

        // solver
        py::class_<Solver >(minstcutModule, sName.c_str(),  solverBase)
            //.def(py::init<>())
        ;

        return settingsCls;

    }

} // namespace minstcut
} // namespace opt
}
}



