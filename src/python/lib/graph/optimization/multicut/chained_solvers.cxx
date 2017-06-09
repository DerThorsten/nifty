#include "pybind11/pybind11.h"
#include "pybind11/stl.h"



// concrete solvers for concrete factories
#include "nifty/graph/optimization/multicut/chained_solvers.hxx"


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
namespace optimization{
namespace multicut{    
    template<class OBJECTIVE>
    void exportChainedSolversT(py::module & multicutModule){


        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::optimization::SolverDocstringHelper docHelper;
        docHelper.objectiveName = "multicut objective"; 
        docHelper.objectiveClsName = MulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name = "chained solvers "; 
        docHelper.mainText =  
            "Chain multiple solvers\n"
            "such that each successor is warm-started with \n"
            "its predecessor solver.\n";
        docHelper.note = "The solvers should be able to be warm started. (Except the first one)";
        



        typedef OBJECTIVE ObjectiveType;
        typedef ChainedSolvers<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        typedef MulticutFactory<Solver> Factory;
        const auto solverName = std::string("ChainedSolvers");
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str(), docHelper)
            .def(py::init<>())
            .def_readwrite("multicutFactories", &Settings::multicutFactories)

        ; 
    }

    
    void exportChainedSolvers(py::module & multicutModule){

        py::options options;
        options.disable_function_signatures();
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportChainedSolversT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportChainedSolversT<ObjectiveType>(multicutModule);
        }    
         
    }
} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
