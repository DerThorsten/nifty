#include <pybind11/pybind11.h>

#include "nifty/graph/opt/mincut/mincut_objective.hxx"
#include "nifty/graph/opt/mincut/mincut_greedy_additive.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/mincut/mincut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/opt/mincut/export_mincut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace mincut{


    template<class OBJECTIVE>
    void exportMincutGreedyAdditiveT(py::module & mincutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef MincutGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        
        exportMincutSolver<Solver>(mincutModule,"MincutGreedyAdditive")
            .def(py::init<>())
            .def_readwrite("nodeNumStopCond", &SettingsType::nodeNumStopCond)
            .def_readwrite("weightStopCond", &SettingsType::weightStopCond)
            .def_readwrite("improve", &SettingsType::improve)
            //.def_readwrite("verbose", &SettingsType::verbose)
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


} // namespace nifty::graph::opt::mincut
} // namespace nifty::graph::opt
}
}
