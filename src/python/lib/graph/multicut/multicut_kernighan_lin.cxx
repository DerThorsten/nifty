#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_kernighan_lin.hxx"

#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{

    template<class OBJECTIVE>
    void exportMulticutKernighanLinT(py::module & multicutModule){
        
        typedef OBJECTIVE ObjectiveType;
        typedef MulticutKernighanLin<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;

        const auto solverName = std::string("MulticutKernighanLin");
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("verbose",&Settings::verbose)
            .def_readwrite("numberOfInnerIterations",&Settings::numberOfInnerIterations)
            .def_readwrite("numberOfOuterIterations",&Settings::numberOfOuterIterations)
            .def_readwrite("epsilon",&Settings::epsilon)
        ;
    }

    void exportMulticutKernighanLin(py::module & multicutModule){
        
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutKernighanLinT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutKernighanLinT<ObjectiveType>(multicutModule);
        }     

    }
}
}
