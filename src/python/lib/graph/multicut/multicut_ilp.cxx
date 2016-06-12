#include <pybind11/pybind11.h>
#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"

// concrete solvers for concrete factories
#include "nifty/graph/multicut/multicut_ilp.hxx"

#ifdef WITH_GUROBI
#include "nifty/graph/multicut/ilp_backend/gurobi.hxx"
#endif

#ifdef WITH_CPLEX
#include "nifty/graph/multicut/ilp_backend/cplex.hxx"
#endif


#include "../../converter.hxx"
#include "py_multicut_factory.hxx"
#include "py_multicut_base.hxx"



namespace py = pybind11;

//PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{




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

}
}
