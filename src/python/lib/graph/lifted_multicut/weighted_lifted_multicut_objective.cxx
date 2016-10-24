#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream> 
#include <pybind11/numpy.h>

#include "nifty/python/graph/optimization/lifted_multicut/weighted_lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_objective_api.hxx"
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

        // standart api
        exportLiftedMulticutObjectiveApi<ObjectiveType>(liftedMulticutObjectiveCls);
        
        // factory
        liftedMulticutModule.def("weightedLiftedMulticutObjective",
            [](const Graph & graph, const uint64_t numberOfWeights, nifty::marray::PyView<uint64_t, 2> uvIds){

                auto obj = new ObjectiveType(graph, numberOfWeights, uvIds);
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("graph"),
            py::arg("numberOfWeights"),
            py::arg("uvIds")
        );



        // not api, special to weightedObj.
        liftedMulticutObjectiveCls
            .def("addWeightedFeatures",
                [](
                    ObjectiveType & self,
                    nifty::marray::PyView<uint64_t, 2> uvIds,
                    nifty::marray::PyView<float, 2> features,
                    nifty::marray::PyView<uint64_t, 1> weightIds
                ){
                    NIFTY_CHECK_OP(uvIds.shape(0), == , features.shape(0),"uvIds has wrong shape");
                    NIFTY_CHECK_OP(uvIds.shape(1), == , 2,"uvIds has wrong shape");
                    NIFTY_CHECK_OP(features.shape(1), == , weightIds.shape(0),"weightIds and feature shape mismatch");


                    std::vector<uint64_t> wi(weightIds.begin(), weightIds.end());
                    std::vector<float> fBuffer(features.shape(0));


                    for(auto c=0; c<uvIds.shape(0); ++c){
                        const auto u = uvIds(c,0);
                        const auto v = uvIds(c,1);

                        for(auto w=0; w<weightIds.size(); ++w){
                            fBuffer[w] = features(c,  w);
                        }
                        self.addWeightedFeatures(u, v, weightIds.begin(), weightIds.end(), fBuffer.begin(), 0);
                    }
                }
                ,
                py::arg("uvIds"),
                py::arg("features"),
                py::arg("weightIds")
            )

        ;


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


            .def("addWeightedFeatures",
                [](
                    ObjectiveType & self,
                    nifty::marray::PyView<float, 2> uvIds,
                    nifty::marray::PyView<float, 2> features,
                    nifty::marray::PyView<float, 1> constTerm,
                    nifty::marray::PyView<uint64_t, 1> weightIds,
                    const bool overwriteConstTerms
                ){
                    NIFTY_CHECK_OP(uvIds.shape(0), == , features.shape(0),"uvIds has wrong shape");
                    NIFTY_CHECK_OP(constTerm.shape(0), == , features.shape(0),"constTerm has wrong shape");
                    NIFTY_CHECK_OP(uvIds.shape(1), == , 2,"uvIds has wrong shape");
                    NIFTY_CHECK_OP(features.shape(1), == , weightIds.shape(0),"weightIds and feature shape mismatch");

                    std::vector<uint64_t> wi(weightIds.begin(), weightIds.end());
                    std::vector<float> fBuffer(features.shape(0));


                    for(auto c=0; c<uvIds.shape(0); ++c){
                        const auto u = uvIds(c,0);
                        const auto v = uvIds(c,1);

                        for(auto w=0; w<weightIds.size(); ++w){
                            fBuffer[w] = features(c,  w);
                        }
                        self.addWeightedFeatures(u, v, weightIds.begin(), weightIds.end(), fBuffer.begin(),constTerm(c),overwriteConstTerms);
                    }
                }
                , 
                py::arg("uvIds"),
                py::arg("features"),
                py::arg("constTerms"),
                py::arg("weightIds"),
                py::arg_t<bool>("overwriteConstTerms",false)
            )
            .def("addWeightedFeature", &ObjectiveType::addWeightedFeature,
                py::arg("u"),
                py::arg("v"),
                py::arg("weightIndex"),
                py::arg("feature")
            )
            .def("addConstTerm", &ObjectiveType::addConstTerm,
                py::arg("u"),
                py::arg("v"),
                py::arg("constTerm")
            )
            .def("setConstTerm", &ObjectiveType::setConstTerm,
                py::arg("u"),
                py::arg("v"),
                py::arg("constTerm")
            )


            .def("getGradient",
                []
                (
                    const ObjectiveType & obj,
                    nifty::marray::PyView<uint64_t, 1> nodeLabels
                ){
                    nifty::marray::PyView<float> g({size_t(obj.numberOfWeights())});
                    obj.getGradient(nodeLabels, g);
                    return g;
                }
            )

            // .def("getWeightedEdge",[]
            //     (
            //         const ObjectiveType & self,
            //         const uint64_t edge
            //     ){
                    

            //         const auto & weightedEdge = self.weightedEdgeCosts()[edge];
            //         for(const auto p : weightedEdge.indexFeatureMap()){
            //             std::cout<<p.first<<" "<<p.second<<"\n";
            //         }
            //         std::cout<<weightedEdge.constTerm()<<"\n";
    
            //     }
            // )
        ;



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
