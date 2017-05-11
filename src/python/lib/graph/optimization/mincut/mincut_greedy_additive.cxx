#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/mincut/mincut_objective.hxx"
#include "nifty/graph/optimization/mincut/mincut_greedy_additive.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/mincut/mincut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/mincut/export_mincut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace mincut{


    template<class OBJECTIVE>
    void exportMincutGreedyAdditiveT(py::module & mincutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef MincutGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        exportMincutSolver<Solver>(mincutModule,"MincutGreedyAdditive")
            .def(py::init<>())
            .def_readwrite("nodeNumStopCond", &Settings::nodeNumStopCond)
            .def_readwrite("weightStopCond", &Settings::weightStopCond)
            .def_readwrite("improve", &Settings::improve)
            //.def_readwrite("verbose", &Settings::verbose)
        ;
     
    }

    void exportMincutGreedyAdditive(py::module & mincutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutGreedyAdditiveT<ObjectiveType>(mincutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutGreedyAdditiveT<ObjectiveType>(mincutModule);
        }
    }


}
}
}
