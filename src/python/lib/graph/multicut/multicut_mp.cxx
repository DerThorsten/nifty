#ifdef WITH_LP_MP

#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_mp.hxx"

#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{

    // TODO exports with different primal solvers once implemented
    
    void exportMpSettings(py::module & multicutModule){
        
        py::class_<MpSettings>(multicutModule, "MpSettings")
            .def(py::init<>())
            .def_readwrite("primalComputationInterval",&MpSettings::primalComputationInterval)
            .def_readwrite("standardReparametrization",&MpSettings::standardReparametrization)
            .def_readwrite("roundingReparametrization",&MpSettings::roundingReparametrization)
            .def_readwrite("tightenReparametrization",&MpSettings::tightenReparametrization)
            .def_readwrite("tighten",&MpSettings::tighten)
            .def_readwrite("tightenInterval",&MpSettings::tightenInterval)
            .def_readwrite("tightenIteration",&MpSettings::tightenIteration)
            .def_readwrite("tightenSlope",&MpSettings::tightenSlope)
            .def_readwrite("tightenConstraintsPercentage",&MpSettings::tightenConstraintsPercentage)
            .def_readwrite("maxIter",&MpSettings::maxIter)
            .def_readwrite("minDualImprovement",&MpSettings::minDualImprovement)
            .def_readwrite("timeout",&MpSettings::timeout)
        ;

    }

    
    template<class OBJECTIVE>
    void exportMulticutMpT(py::module & multicutModule){
        
        typedef OBJECTIVE ObjectiveType;
        typedef MulticutMp<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        // FIXME nIter and verbose have no effect yet
        const auto solverName = std::string("MulticutMp");
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("numberOfIterations", &Settings::numberOfIterations)
            .def_readwrite("verbose",   &Settings::verbose)
            .def_readwrite("mpSettings",&Settings::mpSettings)
        ; 

    }

    
    void exportMulticutMp(py::module & multicutModule){
        
        exportMpSettings(multicutModule);
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutMpT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutMpT<ObjectiveType>(multicutModule);
        }     

    }

} // namespace graph
} // namespace nifty
#endif
