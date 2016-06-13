#pragma once
#ifndef NIFTY_PYTHON_GRAPH_MULTICUT_EXPORT_MULTICUT_SOLVER_HXX
#define NIFTY_PYTHON_GRAPH_MULTICUT_EXPORT_MULTICUT_SOLVER_HXX



#include <pybind11/pybind11.h>

#include "nifty/graph/multicut/multicut_objective.hxx"
#include "py_multicut_factory.hxx"
#include "py_multicut_base.hxx"



namespace py = pybind11;

//PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{


    template<class SOLVER>
    py::class_<typename SOLVER::Settings>  exportMulticutSolver(
        py::module & multicutModule,
        const std::string & solverName,
        const std::string & graphName
    ){

        typedef SOLVER Solver;
        typedef typename Solver::Settings Settings;
        typedef MulticutFactory<Solver> Factory;


        const std::string factoryBaseName = std::string("MulticutFactoryBase")+graphName;
        const std::string solverBaseName = std::string("MulticutBase")+graphName;
        const std::string settingsName = solverName + std::string("Settings")+graphName;
        const std::string factoryName = solverName + std::string("Factory")+graphName;



        py::object factoryBase = multicutModule.attr(factoryBaseName.c_str());
        py::object solverBase = multicutModule.attr(solverBaseName.c_str());

        // settings
        auto settingsCls = py::class_< Settings >(multicutModule, settingsName.c_str())
        ;

        // factory
        py::class_<Factory>(multicutModule, factoryName.c_str(),  factoryBase)
            .def(py::init<const Settings &>(),
                py::arg_t<Settings>("setttings",Settings())
            )
        ;

        // solver
        py::class_<Solver >(multicutModule, solverName.c_str(),  solverBase)
            //.def(py::init<>())
        ;

        return settingsCls;

    }


    /*

    void exportMulticutIlp(py::module & multicutModule) {


        py::object factoryBase = multicutModule.attr("MulticutFactoryBaseUndirectedGraph");
        py::object solverBase = multicutModule.attr("MulticutBaseUndirectedGraph");

        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> Objective;
        typedef PyMulticutFactoryBase<Objective> PyMcFactoryBase;
        typedef MulticutFactoryBase<Objective> McFactoryBase;

        typedef PyMulticutBase<Objective> PyMcBase;
        typedef MulticutBase<Objective> McBase;

        { // scope for name reusing
        #ifdef WITH_CPLEX


            
            typedef ilp_backend::Cplex IlpSolver;
            typedef MulticutIlp<Objective, IlpSolver> Solver;
            typedef typename Solver::Settings Settings;
            typedef MulticutFactory<Solver> Factory;

            // settings
            py::class_< Settings >(multicutModule, "MulticutIlpCplexSettingsUndirectedGraph")
                .def(py::init<>())
                .def_readwrite("numberOfIterations", &Settings::numberOfIterations)
                .def_readwrite("verbose", &Settings::verbose)
                .def_readwrite("verboseIlp", &Settings::verboseIlp)
                .def_readwrite("addThreeCyclesConstraints", &Settings::addThreeCyclesConstraints)
                .def_readwrite("addOnlyViolatedThreeCyclesConstraints", &Settings::addOnlyViolatedThreeCyclesConstraints)

            ;

            // solver
            py::class_<Solver,std::shared_ptr<McBase> >(multicutModule, "MulticutIlpCplexUndirectedGraph",  solverBase)
                //.def(py::init<>())
            ;

            // factory
            py::class_<Factory>(multicutModule, "MulticutIlpCplexFactoryUndirectedGraph",  factoryBase)
                .def(py::init<const Settings &>(),
                    py::arg_t<Settings>("setttings",Settings())
                )
            ;





        #endif
        }
    }

    */

}
}



#endif /* NIFTY_PYTHON_GRAPH_MULTICUT_EXPORT_MULTICUT_SOLVER_HXX */