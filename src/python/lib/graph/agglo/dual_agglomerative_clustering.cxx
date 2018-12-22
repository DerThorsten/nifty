#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"


#include "nifty/tools/runtime_check.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/agglo/export_agglomerative_clustering.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/agglo/agglomerative_clustering.hxx"


#include "nifty/graph/agglo/cluster_policies/dual_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/detail/merge_rules.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


    template<class GRAPH, class ACC_0, class ACC_1,bool WITH_UCM>
    void exportDualClusterPolicyTT(py::module & aggloModule) {
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1>   PyViewFloat1;
        typedef xt::pytensor<uint8_t, 1> PyViewUInt8_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {
            // name and type of cluster operator
            typedef DualClusterPolicy<GraphType, ACC_0, ACC_1, WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("DualClusterPolicy") +  withUcmStr;
            const auto clusterPolicyBaseName2 = clusterPolicyBaseName + ACC_0::staticName() + ACC_1::staticName();
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName + ACC_0::staticName() + ACC_1::staticName();
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                //.def_property_readonly("mergePrios", &ClusterPolicyType::mergePrios)
                //.def_property_readonly("notMergePrios", &ClusterPolicyType::notMergePrios)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & mergePrios,
                    const PyViewFloat1 & notMergePrios,
                    const PyViewUInt8_1 & isLocalEdge,
                    const PyViewFloat1 & edgeSizes,
                    const typename ClusterPolicyType::Acc0SettingsType updateRule0,
                    const typename ClusterPolicyType::Acc1SettingsType updateRule1,
                    const bool zeroInit,
                    const double stopPriority,
                    const uint64_t numberOfNodesStop
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.updateRule0 = updateRule0;
                    s.updateRule1 = updateRule1;
                    s.stopPriority = stopPriority;
                    s.zeroInit = zeroInit;
                    auto ptr = new ClusterPolicyType(graph, mergePrios, notMergePrios, isLocalEdge, edgeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("mergePrios"),
                py::arg("notMergePrios"),
                py::arg("isMergeEdge"),
                py::arg("edgeSizes"),
                py::arg("updateRule0"),
                py::arg("updateRule1"),
                py::arg("zeroInit") = false,
                py::arg("stopPriority") = 0.5,
                py::arg("numberOfNodesStop") = 1
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName2);
        }
    }


    //emplate<class GRAPH, class ACC_0, class ACC_1>
    //oid exportDualClusterPolicyT(py::module & aggloModule) {
    //   exportDualClusterPolicyTT<GRAPH, ACC_0, ACC_1, false>(aggloModule);
    //   //exportfixationClusterPolicy<GRAPH, ACC_0, ACC_1, true >(aggloModule);
    //



    template<class GRAPH, class ACC_0>
    void exportDualClusterPolicyOuter(py::module & aggloModule) {
        typedef GRAPH GraphType;

        typedef merge_rules::ArithmeticMeanEdgeMap<GraphType, double >  ArithmeticMeanAcc;
        typedef merge_rules::GeneralizedMeanEdgeMap<GraphType, double > GeneralizedMeanAcc;
        typedef merge_rules::SmoothMaxEdgeMap<GraphType, double >       SmoothMaxAcc;
        typedef merge_rules::RankOrderEdgeMap<GraphType, double >       RankOrderAcc;
        typedef merge_rules::MaxEdgeMap<GraphType, double >             MaxAcc;
        typedef merge_rules::MinEdgeMap<GraphType, double >             MinAcc;

        exportDualClusterPolicyTT<GraphType, ACC_0, ArithmeticMeanAcc,  false>(aggloModule);
        exportDualClusterPolicyTT<GraphType, ACC_0, GeneralizedMeanAcc, false>(aggloModule);
        exportDualClusterPolicyTT<GraphType, ACC_0, SmoothMaxAcc,       false>(aggloModule);
        exportDualClusterPolicyTT<GraphType, ACC_0, RankOrderAcc,       false>(aggloModule);
        exportDualClusterPolicyTT<GraphType, ACC_0, MaxAcc,             false>(aggloModule);
        exportDualClusterPolicyTT<GraphType, ACC_0, MinAcc,             false>(aggloModule);
    }


    void exportDualAgglomerativeClustering(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;



            typedef merge_rules::ArithmeticMeanEdgeMap<GraphType, double >  ArithmeticMeanAcc;
            typedef merge_rules::GeneralizedMeanEdgeMap<GraphType, double > GeneralizedMeanAcc;
            typedef merge_rules::SmoothMaxEdgeMap<GraphType, double >       SmoothMaxAcc;
            typedef merge_rules::RankOrderEdgeMap<GraphType, double >       RankOrderAcc;
            typedef merge_rules::MaxEdgeMap<GraphType, double >             MaxAcc;
            typedef merge_rules::MinEdgeMap<GraphType, double >             MinAcc;


            exportDualClusterPolicyOuter<GraphType, ArithmeticMeanAcc >(aggloModule);
            exportDualClusterPolicyOuter<GraphType, SmoothMaxAcc      >(aggloModule);
            exportDualClusterPolicyOuter<GraphType, GeneralizedMeanAcc>(aggloModule);
            exportDualClusterPolicyOuter<GraphType, RankOrderAcc      >(aggloModule);
            exportDualClusterPolicyOuter<GraphType, MaxAcc            >(aggloModule);
            exportDualClusterPolicyOuter<GraphType, MinAcc            >(aggloModule);


        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
