#include <pybind11/pybind11.h>

#include "nifty/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/graph/opt/multicut/multicut_greedy_fixation.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"

#include "nifty/python/graph/opt/solver_docstring.hxx"
#include "nifty/python/graph/opt/multicut/export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    template<class OBJECTIVE>
    void exportMulticutGreedyFixationT(py::module & multicutModule) {


        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::opt::SolverDocstringHelper docHelper;
        docHelper.objectiveName =
            "multicut objective";
        docHelper.objectiveClsName =
            MulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name =
            "greedy fixation";
        docHelper.mainText =
            "Find approximate solutions via\n"
            "agglomerative clustering with cannot link constraints.\n";
        docHelper.note =
            "This solver should be used to\n"
            "warm start other solvers with.\n"
            "This solver is very fast but\n"
            "yields rather suboptimal results.\n";

        typedef OBJECTIVE ObjectiveType;
        typedef MulticutGreedyFixation<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;

        exportMulticutSolver<Solver>(multicutModule,"MulticutGreedyFixation",docHelper)
            .def(py::init<>())
            .def_readwrite("nodeNumStopCond", &SettingsType::nodeNumStopCond)
            .def_readwrite("weightStopCond", &SettingsType::weightStopCond)
            .def_readwrite("visitNth", &SettingsType::visitNth)
            //.def_readwrite("verbose", &SettingsType::verbose)
        ;

    }

    void exportMulticutGreedyFixation(py::module & multicutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutGreedyFixationT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutGreedyFixationT<ObjectiveType>(multicutModule);
        }
    }

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
