#pragma once



#include <pybind11/pybind11.h>

#include "nifty/python/graph/opt/mincut/mincut_objective.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"
#include "py_mincut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace opt{
namespace mincut{

    template<class SOLVER>
    py::class_<typename SOLVER::SettingsType>  exportMincutSolver(
        py::module & mincutModule,
        const std::string & solverName
    ){

        typedef SOLVER Solver;
        typedef typename Solver::ObjectiveType ObjectiveType;
        typedef typename Solver::SettingsType SettingsType;
        typedef nifty::graph::opt::common::SolverFactory<Solver> Factory;


        const auto objName = MincutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("SolverFactoryBase")+objName;
        const std::string solverBaseName = std::string("MincutBase") + objName;
        
        const std::string sName = solverName + objName;
        const std::string settingsName = solverName + std::string("SettingsType") + objName;
        const std::string factoryName = solverName + std::string("Factory") + objName;
        std::string factoryFactoryName = factoryName;
        factoryFactoryName[0] = std::tolower(factoryFactoryName[0]);


        py::object factoryBase = mincutModule.attr(factoryBaseName.c_str());
        py::object solverBase = mincutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< SettingsType >(mincutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(mincutModule, factoryName.c_str(),  factoryBase)
            .def(py::init<const SettingsType &>(),
                py::arg_t<SettingsType>("setttings",SettingsType())
            )
        ;

        // solver
        py::class_<Solver >(mincutModule, sName.c_str(),  solverBase)
            //.def(py::init<>())
        ;

        return settingsCls;

    }

} // namespace mincut
} // namespace opt
}
}



