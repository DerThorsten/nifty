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
#include "export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
    



    void exportMulticutIlp(py::module & multicutModule) {



        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> Objective;



        { // scope for name reusing
        #ifdef WITH_CPLEX


            
            typedef ilp_backend::Cplex IlpSolver;
            typedef MulticutIlp<Objective, IlpSolver> Solver;
            typedef typename Solver::Settings Settings;
            typedef MulticutFactory<Solver> Factory;

            exportMulticutSolver<Solver>(multicutModule,"MulticutIlpCplex","UndirectedGraph")
                .def(py::init<>())
                .def_readwrite("numberOfIterations", &Settings::numberOfIterations)
                .def_readwrite("verbose", &Settings::verbose)
                .def_readwrite("verboseIlp", &Settings::verboseIlp)
                .def_readwrite("addThreeCyclesConstraints", &Settings::addThreeCyclesConstraints)
                .def_readwrite("addOnlyViolatedThreeCyclesConstraints", &Settings::addOnlyViolatedThreeCyclesConstraints)
            ;
        #endif
        }
        { // scope for name reusing
        #ifdef WITH_GUROBI


            
            typedef ilp_backend::Gurobi IlpSolver;
            typedef MulticutIlp<Objective, IlpSolver> Solver;
            typedef typename Solver::Settings Settings;
            typedef MulticutFactory<Solver> Factory;

            exportMulticutSolver<Solver>(multicutModule,"MulticutIlpGurobi","UndirectedGraph")
                .def(py::init<>())
                .def_readwrite("numberOfIterations", &Settings::numberOfIterations)
                .def_readwrite("verbose", &Settings::verbose)
                .def_readwrite("verboseIlp", &Settings::verboseIlp)
                .def_readwrite("addThreeCyclesConstraints", &Settings::addThreeCyclesConstraints)
                .def_readwrite("addOnlyViolatedThreeCyclesConstraints", &Settings::addOnlyViolatedThreeCyclesConstraints)
            ;
        #endif
        }


    }

}
}
