#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_greedy_additive.hxx"

#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/weighted_lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/loss_augmented_view_lifted_multicut_objective.hxx"

#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace lifted_multicut{


    template<class OBJECTIVE>
    void exportLiftedMulticutGreedyAdditiveT(py::module & liftedMulticutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule,"LiftedMulticutGreedyAdditive")
            .def(py::init<>())
            .def_readwrite("makeMonoton",     &Settings::makeMonoton)
            .def_readwrite("nodeNumStopCond", &Settings::nodeNumStopCond)
            .def_readwrite("weightStopCond",  &Settings::weightStopCond)
            //.def_readwrite("verbose", &Settings::verbose)
        ;
     
    }

    void exportLiftedMulticutGreedyAdditive(py::module & liftedMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> ObjectiveType;
            exportLiftedMulticutGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> WeightedObjectiveType;
            typedef LossAugmentedViewLiftedMulticutObjective<WeightedObjectiveType> ObjectiveType;
            exportLiftedMulticutGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }

    }

}
}
}
