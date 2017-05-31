#include "pybind11/pybind11.h"
#include "pybind11/stl.h"



// concrete solvers for concrete factories
#include "nifty/graph/optimization/multicut/block_multicut.hxx"


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
    void exportBlockMulticutT(py::module & multicutModule){
        typedef OBJECTIVE ObjectiveType;
        typedef BlockMulticut<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        typedef MulticutFactory<Solver> Factory;
        const auto solverName = std::string("BlockMulticut");
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("multicutFactory", &Settings::multicutFactory)

        ; 
    }

    
    void exportBlockMulticut(py::module & multicutModule){

        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportBlockMulticutT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportBlockMulticutT<ObjectiveType>(multicutModule);
        }    
         
    }
} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
