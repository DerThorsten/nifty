#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/graph/optimization/multicut/multicut_greedy_additive.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

    template<class OBJECTIVE>
    void exportMulticutGreedyAdditiveT(py::module & multicutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef MulticutGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        exportMulticutSolver<Solver>(multicutModule,"MulticutGreedyAdditive")
            .def(py::init<>())
            .def_readwrite("nodeNumStopCond", &Settings::nodeNumStopCond)
            .def_readwrite("weightStopCond", &Settings::weightStopCond)
            //.def_readwrite("verbose", &Settings::verbose)
        ;
     
    }

    void exportMulticutGreedyAdditive(py::module & multicutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutGreedyAdditiveT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutGreedyAdditiveT<ObjectiveType>(multicutModule);
        }
    }

} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
