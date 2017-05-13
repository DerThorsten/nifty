#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_andres.hxx"

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
    void exportMulticutAndresT(py::module & multicutModule){
        
        typedef OBJECTIVE ObjectiveType;

        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        
        // export greedy additive
        {
            typedef MulticutAndresGreedyAdditive<ObjectiveType> Solver;
            typedef typename Solver::Settings Settings;
            const auto solverName = std::string("MulticutAndresGreedyAdditive");
            // FIXME verbose has no effect yet
            exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
                .def(py::init<>())
            ; 
        }
        
        // export kernighan lin
        {
            typedef MulticutAndresKernighanLin<ObjectiveType> Solver;
            typedef typename Solver::Settings Settings;
            const auto solverName = std::string("MulticutAndresKernighanLin");
            // FIXME verbose has no effect yet
            exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
                .def(py::init<>())
                .def_readwrite("numberOfInnerIterations", &Settings::numberOfInnerIterations)
                .def_readwrite("numberOfOuterIterations", &Settings::numberOfOuterIterations)
                .def_readwrite("epsilon", &Settings::epsilon)
                .def_readwrite("verbose", &Settings::verbose)
                .def_readwrite("greedyWarmstart", &Settings::greedyWarmstart)
            ; 
        }

    }

    
    void exportMulticutAndres(py::module & multicutModule){
        
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutAndresT<ObjectiveType>(multicutModule);
        }
        // FIXME this doesn't compile
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportMulticutMpT<ObjectiveType>(multicutModule);
        //}     

    }

} // namespace graph
} // namespace nifty
