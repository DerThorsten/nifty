#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream> 
#include <pybind11/numpy.h>


#include "nifty/python/graph/optimization/lifted_multicut/weighted_lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/loss_augmented_view_lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_objective_api.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class GRAPH>
    void exportLossAugmentedViewLiftedMulticutObjectiveT(py::module & liftedMulticutModule) {

        typedef GRAPH Graph;
        typedef WeightedLiftedMulticutObjective<Graph, float> WeightedObjectiveType;
        typedef LossAugmentedViewLiftedMulticutObjective<WeightedObjectiveType> ObjectiveType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;


        const auto clsName = LiftedMulticutObjectiveName<ObjectiveType>::name();


        auto liftedMulticutObjectiveCls = py::class_<ObjectiveType>(
            liftedMulticutModule, clsName.c_str()
        );

        // standart api
        exportLiftedMulticutObjectiveApi<ObjectiveType>(liftedMulticutObjectiveCls);
        
        //factory
        liftedMulticutModule.def("lossAugmentedViewLiftedMulticutObjective",
            []
            (
                WeightedObjectiveType & objective,
                nifty::marray::PyView<uint64_t, 1> nodeGroundTruth,
                nifty::marray::PyView<float, 1>    loss
            ){
                auto obj = new ObjectiveType(objective,nodeGroundTruth,loss);
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("objective"),
            py::arg("nodeGroundTruth"),
            py::arg("loss")
        );


        liftedMulticutObjectiveCls

            .def("changeWeights",
                [](
                    ObjectiveType & self,
                    nifty::marray::PyView<float, 1> weights
                ){
                    self.changeWeights(weights);
                }
                , 
                py::arg("weightVector")
            )
        ;



    }

    void exportLossAugmentedViewLiftedMulticutObjective(py::module & liftedMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportLossAugmentedViewLiftedMulticutObjectiveT<GraphType>(liftedMulticutModule);
        }
    }

}
}
}
