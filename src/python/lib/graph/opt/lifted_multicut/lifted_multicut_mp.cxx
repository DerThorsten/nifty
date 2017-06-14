#ifdef WITH_LP_MP

#include <pybind11/pybind11.h>

#include "nifty/python/graph/opt/lifted_multicut/export_lifted_multicut_solver.hxx"
#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/opt/lifted_multicut/lifted_multicut_mp.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{

    template<class OBJECTIVE>
    void exportLiftedMulticutMpT(py::module & liftedMulticutModule) {
        
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutMp<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        const auto solverName = std::string("LiftedMulticutMp");
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("lmcFactory", &SettingsType::lmcFactory)
            .def_readwrite("greedyWarmstart", &SettingsType::greedyWarmstart)
            .def_readwrite("tightenSlope", &SettingsType::tightenSlope)
            .def_readwrite("tightenMinDualImprovementInterval", &SettingsType::tightenMinDualImprovementInterval)
            .def_readwrite("tightenMinDualImprovement", &SettingsType::tightenMinDualImprovement)
            .def_readwrite("tightenConstraintsPercentage", &SettingsType::tightenConstraintsPercentage)
            .def_readwrite("tightenConstraintsMax", &SettingsType::tightenConstraintsMax)
            .def_readwrite("tightenInterval", &SettingsType::tightenInterval)
            .def_readwrite("tightenIteration", &SettingsType::tightenIteration)
            .def_readwrite("tightenReparametrization", &SettingsType::tightenReparametrization)
            .def_readwrite("roundingReparametrization", &SettingsType::roundingReparametrization)
            .def_readwrite("standardReparametrization", &SettingsType::standardReparametrization)
            .def_readwrite("tighten", &SettingsType::tighten)
            .def_readwrite("minDualImprovementInterval", &SettingsType::minDualImprovementInterval)
            .def_readwrite("minDualImprovement", &SettingsType::minDualImprovement)
            .def_readwrite("lowerBoundComputationInterval", &SettingsType::lowerBoundComputationInterval)
            .def_readwrite("primalComputationInterval", &SettingsType::primalComputationInterval)
            .def_readwrite("timeout", &SettingsType::timeout)
            .def_readwrite("maxIter", &SettingsType::maxIter)
            .def_readwrite("numThreads", &SettingsType::numLpThreads)
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
} // namespace nifty::graph::opt
}
}
#endif
