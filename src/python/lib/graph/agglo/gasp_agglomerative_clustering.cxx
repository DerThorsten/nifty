#include <pybind11/pybind11.h>
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/agglo/export_agglomerative_clustering.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/agglo/agglomerative_clustering.hxx"


#include "nifty/graph/agglo/cluster_policies/gasp_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/detail/merge_rules.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
    namespace graph{
        namespace agglo{


            template<class GRAPH, class UPDATE_RULE, bool WITH_UCM>
            void exportGaspClusterPolicyTT(py::module & aggloModule) {

                typedef GRAPH GraphType;
                const auto graphName = GraphName<GraphType>::name();
                typedef xt::pytensor<float, 1>   PyViewFloat1;
                typedef xt::pytensor<double, 1>   PyViewDouble1;
                typedef xt::pytensor<uint8_t, 1> PyViewUInt8_1;
                const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

                {
                    // name and type of cluster operator
                    typedef GaspClusterPolicy<GraphType, UPDATE_RULE, WITH_UCM> ClusterPolicyType;
                    const auto clusterPolicyBaseName = std::string("GaspClusterPolicy") +  withUcmStr;
                    const auto clusterPolicyBaseName2 = clusterPolicyBaseName + UPDATE_RULE::staticName();
                    const auto clusterPolicyClsName = clusterPolicyBaseName + graphName + UPDATE_RULE::staticName();
                    const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

                    // the cluster operator cls
                    auto clusterClass = py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                    ;

                    clusterClass
                            .def("exportAgglomerationData", [](
                                         ClusterPolicyType * self
                                 ){
//                        auto edgeContractionGraph = self->edgeContractionGraph();
                                     auto out1 = self->exportFinalNodeDataOriginalGraph();
                                     auto out2 = self->exportFinalEdgeDataContractedGraph();
                                     auto out3 = self->exportAction();
                                     return std::make_tuple(out1, out2, out3);
                                 }
                            );



                    // factory
                    aggloModule.def(clusterPolicyFacName.c_str(),
                                    [](
                                            const GraphType & graph,
                                            const PyViewDouble1 & signedWeights,
                                            const PyViewUInt8_1 & isLocalEdge,
                                            const PyViewDouble1 & edgeSizes,
                                            const PyViewDouble1 & nodeSizes,
                                            const typename ClusterPolicyType::UpdateRuleSettingsType updateRule,
                                            const uint64_t numberOfNodesStop,
                                            const double sizeRegularizer,
                                            const bool addNonLinkConstraints,
                                            const bool mergeConstrainedEdgesAtTheEnd,
                                            const bool collectStats
                                    ){
                                        typename ClusterPolicyType::SettingsType s;
                                        s.numberOfNodesStop = numberOfNodesStop;
                                        s.sizeRegularizer = sizeRegularizer;
                                        s.updateRule = updateRule;
                                        s.addNonLinkConstraints = addNonLinkConstraints;
                                        s.mergeConstrainedEdgesAtTheEnd = mergeConstrainedEdgesAtTheEnd;
                                        s.collectStats = collectStats;
                                        auto ptr = new ClusterPolicyType(graph, signedWeights, isLocalEdge, edgeSizes, nodeSizes, s);
                                        return ptr;
                                    },
                                    py::return_value_policy::take_ownership,
                                    py::keep_alive<0,1>(), // graph
                                    py::arg("graph"),
                                    py::arg("signedWeights"),
                                    py::arg("isMergeEdge"),
                                    py::arg("edgeSizes"),
                                    py::arg("nodeSizes"),
                                    py::arg("updateRule0"),
                                    py::arg("numberOfNodesStop") = 1,
                                    py::arg("sizeRegularizer") = 0.,
                                    py::arg("addNonLinkConstraints") = false,
                                    py::arg("mergeConstrainedEdgesAtTheEnd") = false,
                                    py::arg("collectStats") = false
                    );

                    // export the agglomerative clustering functionality for this cluster operator
                    exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName2);
                }
            }


            void exportGaspAgglomerativeClustering(py::module & aggloModule) {
                {
                    typedef PyUndirectedGraph GraphType;

                    typedef merge_rules::SumEdgeMap<GraphType, double >  SumAcc;
                    typedef merge_rules::ArithmeticMeanEdgeMap<GraphType, double >  ArithmeticMeanAcc;
                    typedef merge_rules::GeneralizedMeanEdgeMap<GraphType, double > GeneralizedMeanAcc;
                    typedef merge_rules::SmoothMaxEdgeMap<GraphType, double >       SmoothMaxAcc;
                    typedef merge_rules::RankOrderEdgeMap<GraphType, double >       RankOrderAcc;
                    typedef merge_rules::MaxEdgeMap<GraphType, double >             MaxAcc;
                    typedef merge_rules::MinEdgeMap<GraphType, double >             MinAcc;
                    typedef merge_rules::MutexWatershedEdgeMap<GraphType, double >             MWSAcc;


                    exportGaspClusterPolicyTT<GraphType, SumAcc, false >(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, ArithmeticMeanAcc, false >(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, SmoothMaxAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, GeneralizedMeanAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, RankOrderAcc , false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MaxAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MinAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MWSAcc, false>(aggloModule);


                }
                {
                    typedef UndirectedGridGraph<2, true> GraphType;

                    typedef merge_rules::SumEdgeMap<GraphType, double >  SumAcc;
                    typedef merge_rules::ArithmeticMeanEdgeMap<GraphType, double >  ArithmeticMeanAcc;
                    typedef merge_rules::GeneralizedMeanEdgeMap<GraphType, double > GeneralizedMeanAcc;
                    typedef merge_rules::SmoothMaxEdgeMap<GraphType, double >       SmoothMaxAcc;
                    typedef merge_rules::RankOrderEdgeMap<GraphType, double >       RankOrderAcc;
                    typedef merge_rules::MaxEdgeMap<GraphType, double >             MaxAcc;
                    typedef merge_rules::MinEdgeMap<GraphType, double >             MinAcc;
                    typedef merge_rules::MutexWatershedEdgeMap<GraphType, double >             MWSAcc;


                    exportGaspClusterPolicyTT<GraphType, SumAcc, false >(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, ArithmeticMeanAcc, false >(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, SmoothMaxAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, GeneralizedMeanAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, RankOrderAcc , false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MaxAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MinAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MWSAcc, false>(aggloModule);

                }
                {
                    typedef UndirectedGridGraph<3,true> GraphType;

                    typedef merge_rules::SumEdgeMap<GraphType, double >  SumAcc;
                    typedef merge_rules::ArithmeticMeanEdgeMap<GraphType, double >  ArithmeticMeanAcc;
                    typedef merge_rules::GeneralizedMeanEdgeMap<GraphType, double > GeneralizedMeanAcc;
                    typedef merge_rules::SmoothMaxEdgeMap<GraphType, double >       SmoothMaxAcc;
                    typedef merge_rules::RankOrderEdgeMap<GraphType, double >       RankOrderAcc;
                    typedef merge_rules::MaxEdgeMap<GraphType, double >             MaxAcc;
                    typedef merge_rules::MinEdgeMap<GraphType, double >             MinAcc;
                    typedef merge_rules::MutexWatershedEdgeMap<GraphType, double >             MWSAcc;


                    exportGaspClusterPolicyTT<GraphType, SumAcc, false >(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, ArithmeticMeanAcc, false >(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, SmoothMaxAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, GeneralizedMeanAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, RankOrderAcc , false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MaxAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MinAcc, false>(aggloModule);
                    exportGaspClusterPolicyTT<GraphType, MWSAcc, false>(aggloModule);

                }

            }

        } // end namespace agglo
    } // end namespace graph
} // end namespace nifty
