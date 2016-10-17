#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_andres_kernighan_lin.hxx"

#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
//#include "nifty/python/graph/edge_contraction_graph.hxx"
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
    void exportLiftedMulticutAndresKernighanLinT(py::module & liftedMulticutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutAndresKernighanLin<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule,"LiftedMulticutAndresKernighanLin")
            .def(py::init<>())
            .def_readwrite("numberOfInnerIterations", &Settings::numberOfInnerIterations)
            .def_readwrite("numberOfOuterIterations", &Settings::numberOfOuterIterations)
            .def_readwrite("epsilon", &Settings::epsilon)
            //.def_readwrite("numberOfOuterIterations", &Settings::numberOfOuterIterations)

            //.def_readwrite("verbose", &Settings::verbose)
        ;
     
    }

    void exportLiftedMulticutAndresKernighanLin(py::module & liftedMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutAndresKernighanLinT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> ObjectiveType;
            exportLiftedMulticutAndresKernighanLinT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> WeightedObjectiveType;
            typedef LossAugmentedViewLiftedMulticutObjective<WeightedObjectiveType> ObjectiveType;
            exportLiftedMulticutAndresKernighanLinT<ObjectiveType>(liftedMulticutModule);
        }
    }

}
}
}
