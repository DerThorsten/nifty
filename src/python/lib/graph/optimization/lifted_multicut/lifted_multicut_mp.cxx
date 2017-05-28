#ifdef WITH_LP_MP

#include <pybind11/pybind11.h>

#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_solver.hxx"
#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_mp.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class OBJECTIVE>
    void exportLiftedMulticutMpT(py::module & liftedMulticutModule) {
        
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutMp<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        typedef LiftedMulticutFactory<Solver> Factory;
        const auto solverName = std::string("LiftedMulticutMp");
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("lmcFactory", &Settings::lmcFactory)
            .def_readwrite("greedyWarmstart", &Settings::greedyWarmstart)
            .def_readwrite("tightenSlope", &Settings::tightenSlope)
            .def_readwrite("tightenMinDualImprovementInterval", &Settings::tightenMinDualImprovementInterval)
            .def_readwrite("tightenMinDualImprovement", &Settings::tightenMinDualImprovement)
            .def_readwrite("tightenConstraintsPercentage", &Settings::tightenConstraintsPercentage)
            .def_readwrite("tightenConstraintsMax", &Settings::tightenConstraintsMax)
            .def_readwrite("tightenInterval", &Settings::tightenInterval)
            .def_readwrite("tightenIteration", &Settings::tightenIteration)
            .def_readwrite("tightenReparametrization", &Settings::tightenReparametrization)
            .def_readwrite("roundingReparametrization", &Settings::roundingReparametrization)
            .def_readwrite("standardReparametrization", &Settings::standardReparametrization)
            .def_readwrite("tighten", &Settings::tighten)
            .def_readwrite("minDualImprovementInterval", &Settings::minDualImprovementInterval)
            .def_readwrite("minDualImprovement", &Settings::minDualImprovement)
            .def_readwrite("lowerBoundComputationInterval", &Settings::lowerBoundComputationInterval)
            .def_readwrite("primalComputationInterval", &Settings::primalComputationInterval)
            .def_readwrite("timeout", &Settings::timeout)
            .def_readwrite("maxIter", &Settings::maxIter)
            .def_readwrite("numThreads", &Settings::numLpThreads)
        ;
    }
   

    void exportLiftedMulticutMp(py::module & liftedMulticutModule){
        
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType,double> ObjectiveType;
            exportLiftedMulticutMpT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType,double> ObjectiveType;
            exportLiftedMulticutMpT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutIlpT<ObjectiveType>(liftedMulticutModule);
        //}    
    }


}
}
}
#endif
