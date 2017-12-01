#include "pybind11/pybind11.h"
#include "pybind11/stl.h"



// concrete solvers for concrete factories
#include "nifty/graph/opt/lifted_multicut/chained_solvers.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/opt/solver_docstring.hxx"
#include "nifty/python/graph/opt/lifted_multicut/export_lifted_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{    
    template<class OBJECTIVE>
    void exportChainedSolversT(py::module & liftedMulticutModule){


        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::opt::SolverDocstringHelper docHelper;
        docHelper.objectiveName = "lifted_multicut objective"; 
        docHelper.objectiveClsName = LiftedMulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name = "chained solvers "; 
        docHelper.mainText =  
            "Chain multiple solvers\n"
            "such that each successor is warm-started with \n"
            "its predecessor solver.\n";
        docHelper.note = "The solvers should be able to be warm started. (Except the first one)";
        



        typedef OBJECTIVE ObjectiveType;
        typedef ChainedSolvers<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        const auto solverName = std::string("ChainedSolvers");
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("factories", &SettingsType::factories)

        ; 
    }

    
    void exportChainedSolvers(py::module & liftedMulticutModule){

        py::options options;
        options.disable_function_signatures();
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportChainedSolversT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportChainedSolversT<ObjectiveType>(liftedMulticutModule);
        }   
         
    }
} // namespace nifty::graph::opt::lifted_multicut
} // namespace nifty::graph::opt
}
}
