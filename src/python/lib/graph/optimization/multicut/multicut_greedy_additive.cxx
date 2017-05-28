#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/graph/optimization/multicut/multicut_greedy_additive.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"

#include "nifty/python/graph/optimization/solver_docstring.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{



    template<class OBJECTIVE>
    void exportMulticutGreedyAdditiveT(py::module & multicutModule) {


        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::optimization::SolverDocstringHelper docHelper;
        docHelper.objectiveName =
            "multicut objective";
        docHelper.objectiveClsName = 
            MulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name = 
            "greedy additive";
        docHelper.mainText =  
            "Find approximate solutions via\n"
            "agglomerative clustering as in :cite:`beier_15_funsion`.\n";
        docHelper.cites.emplace_back("beier_15_funsion");
        docHelper.note = 
            "This solver should be used to\n"        
            "warm start other solvers with.\n"
            "This solver is very fast but\n"
            "yields rather suboptimal results.\n";



        typedef OBJECTIVE ObjectiveType;
        typedef MulticutGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        exportMulticutSolver<Solver>(multicutModule,"MulticutGreedyAdditive",docHelper)
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

}
}
