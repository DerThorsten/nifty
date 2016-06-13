#include <pybind11/pybind11.h>

#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/multicut/multicut_greedy_additive.hxx"

#include "../../converter.hxx"
#include "export_multicut_solver.hxx"

namespace py = pybind11;

//PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{




    void exportMulticutGreedyAdditive(py::module & multicutModule) {



        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> Objective;
        typedef MulticutGreedyAdditive<Objective> Solver;
        typedef typename Solver::Settings Settings;



        exportMulticutSolver<Solver>(multicutModule,"MulticutGreedyAdditive","UndirectedGraph")
            .def(py::init<>())
            .def_readwrite("nodeNumStopCond", &Settings::nodeNumStopCond)
            .def_readwrite("weightStopCond", &Settings::weightStopCond)
            .def_readwrite("verbose", &Settings::verbose)
        ;
     
    }

}
}
