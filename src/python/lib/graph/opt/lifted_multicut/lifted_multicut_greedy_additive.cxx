#include <pybind11/pybind11.h>

#include "nifty/graph/opt/lifted_multicut/lifted_multicut_greedy_additive.hxx"

#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/opt/lifted_multicut/export_lifted_multicut_solver.hxx"


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{


    template<class OBJECTIVE>
    void exportLiftedMulticutGreedyAdditiveT(py::module & liftedMulticutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule,"LiftedMulticutGreedyAdditive")
            .def(py::init<>())
            .def_readwrite("nodeNumStopCond", &SettingsType::nodeNumStopCond)
            .def_readwrite("weightStopCond", &SettingsType::weightStopCond)
            //.def_readwrite("verbose", &SettingsType::verbose)
        ;
     
    }

    void exportLiftedMulticutGreedyAdditive(py::module & liftedMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<3,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        //}
    }

}
} // namespace nifty::graph::opt
}
}
