#include <pybind11/pybind11.h>



// concrete solvers for concrete factories
#include "nifty/graph/optimization/multicut/cgc.hxx"


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
    
    template<class OBJECTIVE>
    void exportCgcT(py::module & multicutModule){
        typedef OBJECTIVE ObjectiveType;
        typedef Cgc<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        typedef MulticutFactory<Solver> Factory;
        const auto solverName = std::string("Cgc");
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
            .def(py::init<>())

            .def_readwrite("doCutPhase", &Settings::doCutPhase)
            .def_readwrite("doGlueAndCutPhase", &Settings::doGlueAndCutPhase)


        ; 
    }

    
    void exportCgc(py::module & multicutModule){

        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportCgcT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportCgcT<ObjectiveType>(multicutModule);
        }    
         
    }
}
}
