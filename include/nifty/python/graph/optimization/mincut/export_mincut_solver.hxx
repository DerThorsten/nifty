#pragma once
#ifndef NIFTY_PYTHON_GRAPH_MULTICUT_EXPORT_MULTICUT_SOLVER_HXX
#define NIFTY_PYTHON_GRAPH_MULTICUT_EXPORT_MULTICUT_SOLVER_HXX



#include <pybind11/pybind11.h>

#include "nifty/python/graph/optimization/mincut/mincut_objective.hxx"
#include "py_mincut_factory.hxx"
#include "py_mincut_base.hxx"



namespace py = pybind11;




namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{

    template<class SOLVER>
    py::class_<typename SOLVER::Settings>  exportMincutSolver(
        py::module & mincutModule,
        const std::string & solverName
    ){

        typedef SOLVER Solver;
        typedef typename Solver::Objective ObjectiveType;
        typedef typename Solver::Settings Settings;
        typedef MincutFactory<Solver> Factory;
        typedef MincutFactoryBase<ObjectiveType> McFactoryBase;

        const auto objName = MincutObjectiveName<ObjectiveType>::name();

        const std::string factoryBaseName = std::string("MincutFactoryBase")+objName;
        const std::string solverBaseName = std::string("MincutBase") + objName;
        
        const std::string sName = solverName + objName;
        const std::string settingsName = solverName + std::string("Settings") + objName;
        const std::string factoryName = solverName + std::string("Factory") + objName;
        std::string factoryFactoryName = factoryName;
        factoryFactoryName[0] = std::tolower(factoryFactoryName[0]);


        py::object factoryBase = mincutModule.attr(factoryBaseName.c_str());
        py::object solverBase = mincutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< Settings >(mincutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory, std::shared_ptr<Factory> >(mincutModule, factoryName.c_str(),  factoryBase)
            .def(py::init<const Settings &>(),
                py::arg_t<Settings>("setttings",Settings())
            )
        ;

        // solver
        py::class_<Solver >(mincutModule, sName.c_str(),  solverBase)
            //.def(py::init<>())
        ;

        return settingsCls;

    }


    /*

    void exportMincutIlp(py::module & mincutModule) {


        py::object factoryBase = mincutModule.attr("MincutFactoryBaseUndirectedGraph");
        py::object solverBase = mincutModule.attr("MincutBaseUndirectedGraph");

        typedef UndirectedGraph<> Graph;
        typedef MincutObjective<Graph, double> Objective;
        typedef PyMincutFactoryBase<Objective> PyMcFactoryBase;
        typedef MincutFactoryBase<Objective> McFactoryBase;

        typedef PyMincutBase<Objective> PyMcBase;
        typedef MincutBase<Objective> McBase;

        { // scope for name reusing
        #ifdef WITH_CPLEX


            
            typedef ilp_backend::Cplex IlpSolver;
            typedef MincutIlp<Objective, IlpSolver> Solver;
            typedef typename Solver::Settings Settings;
            typedef MincutFactory<Solver> Factory;

            // settings
            py::class_< Settings >(mincutModule, "MincutIlpCplexSettingsUndirectedGraph")
                .def(py::init<>())
                .def_readwrite("numberOfIterations", &Settings::numberOfIterations)
                .def_readwrite("verbose", &Settings::verbose)
                .def_readwrite("verboseIlp", &Settings::verboseIlp)
                .def_readwrite("addThreeCyclesConstraints", &Settings::addThreeCyclesConstraints)
                .def_readwrite("addOnlyViolatedThreeCyclesConstraints", &Settings::addOnlyViolatedThreeCyclesConstraints)

            ;

            // solver
            py::class_<Solver,std::shared_ptr<McBase> >(mincutModule, "MincutIlpCplexUndirectedGraph",  solverBase)
                //.def(py::init<>())
            ;

            // factory
            py::class_<Factory>(mincutModule, "MincutIlpCplexFactoryUndirectedGraph",  factoryBase)
                .def(py::init<const Settings &>(),
                    py::arg_t<Settings>("setttings",Settings())
                )
            ;





        #endif
        }
    }

    */
} // namespace mincut
} // namespace optimization
}
}



#endif /* NIFTY_PYTHON_GRAPH_MULTICUT_EXPORT_MULTICUT_SOLVER_HXX */
