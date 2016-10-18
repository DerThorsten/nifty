#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/structured_learning/learners/struct_max_margin/struct_max_margin.hxx"
#include "nifty/structured_learning/instances/struct_max_margin_oracle_lmc.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/weighted_lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/loss_augmented_view_lifted_multicut_objective.hxx"


namespace py = pybind11;


//PYBIND11_DECLARE_HOLDER_TYPE(SSMOracleBase, std::shared_ptr<SSMOracleBase>);

namespace nifty{
namespace structured_learning{







    void exportStructMaxMarginOracleLmc(py::module & structuredLearningModule){

        typedef nifty::graph::PyUndirectedGraph GraphType;
        typedef nifty::graph::lifted_multicut::WeightedLiftedMulticutObjective<GraphType, float> WeightedObjectiveType;
        typedef nifty::graph::lifted_multicut::LossAugmentedViewLiftedMulticutObjective<WeightedObjectiveType> LossAugemntedObjectiveType;


        py::object oracleBase = structuredLearningModule.attr("StructMaxMarginOracleBase");

        typedef StructMaxMarginOracleLmc<WeightedObjectiveType, LossAugemntedObjectiveType> Oracle;
    

        py::class_<Oracle >(structuredLearningModule, "StructMaxMarginOracleLmc",  oracleBase)
            .def(py::init<const size_t >(),
                py::arg("numberOfWeights")
            )

            .def("addModel",[](
                Oracle * oracle,
                const GraphType & graph,
                nifty::marray::PyView<uint64_t, 1> nodeGroundTruth,
                nifty::marray::PyView<float, 1>    nodeSizes
            ){
                oracle->addWeightedModel(graph, oracle->numberOfWeights());

                // get the just added model
                const auto index = oracle->numberOfWeightesModels() - 1;
                auto & weightedModel = oracle->getWeightedModel(index);


                oracle->addLossAugmentedModel(weightedModel, nodeGroundTruth, nodeSizes);
            })

            .def("getWeightedModel",&Oracle::getWeightedModel, py::return_value_policy::reference_internal)
            .def("getLossAugmentedModel",&Oracle::getLossAugmentedModel, py::return_value_policy::reference_internal)
        ;



    }
}
}