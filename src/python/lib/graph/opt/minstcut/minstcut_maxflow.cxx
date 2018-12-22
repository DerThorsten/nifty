#include <pybind11/pybind11.h>

#include "nifty/graph/opt/minstcut/minstcut_objective.hxx"
#include "nifty/graph/opt/minstcut/minstcut_maxflow.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/minstcut/minstcut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/opt/minstcut/export_minstcut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{


    template<class OBJECTIVE>
    void exportMinstcutMaxflowT(py::module & module) {

        typedef OBJECTIVE ObjectiveType;
        typedef MinstcutMaxflow<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;

        exportMinstcutSolver<Solver>(module, "MinstcutMaxflow")
            .def(py::init<>())
        ;
    }

    void exportMinstcutMaxflow(py::module & module) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            exportMinstcutMaxflowT<ObjectiveType>(module);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            exportMinstcutMaxflowT<ObjectiveType>(module);
        }
    }

} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
}
}
