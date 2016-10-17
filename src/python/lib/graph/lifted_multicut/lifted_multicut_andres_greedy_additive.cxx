#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_andres_greedy_additive.hxx"

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
    void exportLiftedMulticutAndresGreedyAdditiveT(py::module & liftedMulticutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutAndresGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule,"LiftedMulticutAndresGreedyAdditive")
            .def(py::init<>())
        ;
     
    }

    void exportLiftedMulticutAndresGreedyAdditive(py::module & liftedMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutAndresGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> ObjectiveType;
            exportLiftedMulticutAndresGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> WeightedObjectiveType;
            typedef LossAugmentedViewLiftedMulticutObjective<WeightedObjectiveType> ObjectiveType;
            exportLiftedMulticutAndresGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
    }

}
}
}
