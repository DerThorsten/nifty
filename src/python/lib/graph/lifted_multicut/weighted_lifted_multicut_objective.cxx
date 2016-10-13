#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream> 
#include <pybind11/numpy.h>

#include "nifty/python/graph/optimization/lifted_multicut/weighted_lifted_multicut_objective.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class GRAPH>
    void exportWeightedLiftedMulticutObjectiveT(py::module & liftedMulticutModule) {

        typedef GRAPH Graph;
        typedef WeightedLiftedMulticutObjective<Graph, float> ObjectiveType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        const auto clsName = LiftedMulticutObjectiveName<ObjectiveType>::name();


        auto liftedMulticutObjectiveCls = py::class_<ObjectiveType>(
            liftedMulticutModule, clsName.c_str()
        );


        liftedMulticutObjectiveCls
            .def("addWeightedFeatures",
                [](
                    ObjectiveType & self,
                    nifty::marray::PyView<float, 2> uvIds,
                    nifty::marray::PyView<float, 2> features,
                    nifty::marray::PyView<float, 1> weightIds
                ){
                    NIFTY_CHECK_OP(uvIds.shape(0), == , features.shape(0),"uvIds has wrong shape");
                    NIFTY_CHECK_OP(uvIds.shape(1), == , 2,"uvIds has wrong shape");


                }
            )
        ;

        liftedMulticutModule.def("weightedLiftedMulticutObjective",
            [](const Graph & graph){

                auto obj = new ObjectiveType(graph);
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("graph")
        );

    }

    void exportWeightedLiftedMulticutObjective(py::module & liftedMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportWeightedLiftedMulticutObjectiveT<GraphType>(liftedMulticutModule);
        }
    }

}
}
}
