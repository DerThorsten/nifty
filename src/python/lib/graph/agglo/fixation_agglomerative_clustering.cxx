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


#include "nifty/graph/agglo/cluster_policies/generalized_fixation_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/fixation_cluster_policy2.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{





    template<class GRAPH, bool WITH_UCM>
    void exportFixationPolicy2(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
        typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef FixationClusterPolicy2<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("FixationClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                .def_property_readonly("mergePrios", &ClusterPolicyType::mergePrios)
                .def_property_readonly("notMergePrios", &ClusterPolicyType::notMergePrios)
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
                    const uint64_t numberOfNodesStop
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
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
                py::arg("numberOfNodesStop") = 1
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    // template<class GRAPH, bool WITH_UCM>
    // void exportFixationPolicy3(py::module & aggloModule) {
        
    //     typedef GRAPH GraphType;
    //     const auto graphName = GraphName<GraphType>::name();
    //     typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
    //     typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
    //     const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

    //     {   
    //         // name and type of cluster operator
    //         typedef FixationClusterPolicy3<GraphType,WITH_UCM> ClusterPolicyType;
    //         const auto clusterPolicyBaseName = std::string("FixationClusterPolicy3") +  withUcmStr;
    //         const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
    //         const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

    //         // the cluster operator cls
    //         py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
    //             .def_property_readonly("mergePrios", &ClusterPolicyType::mergePrios)
    //             .def_property_readonly("notMergePrios", &ClusterPolicyType::notMergePrios)
    //             //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
    //         ;
        

    //         // factory
    //         aggloModule.def(clusterPolicyFacName.c_str(),
    //             [](
    //                 const GraphType & graph,
    //                 const PyViewFloat1 & mergePrios,
    //                 const PyViewFloat1 & notMergePrios,
    //                 const PyViewUInt8_1 & isMergeEdge,
    //                 const PyViewFloat1 & edgeSizes,
    //                 const uint64_t numberOfNodesStop
    //             ){
    //                 typename ClusterPolicyType::SettingsType s;
    //                 s.numberOfNodesStop = numberOfNodesStop;
    //                 auto ptr = new ClusterPolicyType(graph, mergePrios, notMergePrios, isMergeEdge, edgeSizes, s);
    //                 return ptr;
    //             },
    //             py::return_value_policy::take_ownership,
    //             py::keep_alive<0,1>(), // graph
    //             py::arg("graph"),
    //             py::arg("mergePrios"),
    //             py::arg("notMergePrios"),
    //             py::arg("isMergeEdge"),
    //             py::arg("edgeSizes"),
    //             py::arg("numberOfNodesStop") = 1
    //         );

    //         // export the agglomerative clustering functionality for this cluster operator
    //         exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
    //     }
    // }

    template<class GRAPH, bool WITH_UCM>
    void exportGeneralizedFixationPolicy(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
        typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef GeneralizedFixationClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("GeneralizedFixationClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                .def_property_readonly("mergePrios", &ClusterPolicyType::mergePrios)
                .def_property_readonly("notMergePrios", &ClusterPolicyType::notMergePrios)
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
                    const double p0,
                    const double p1,
                    const uint64_t numberOfNodesStop
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.p0 = p0;
                    s.p1 = p1;
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
                py::arg("p0") = 1.0,
                py::arg("p1") = 1.0,
                py::arg("numberOfNodesStop") = 1
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }

    void exportFixationAgglomerativeClustering(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;


            exportFixationPolicy2<GraphType, false>(aggloModule);
            exportFixationPolicy2<GraphType, true>(aggloModule);

            // exportFixationPolicy3<GraphType, false>(aggloModule);
            // exportFixationPolicy3<GraphType, true>(aggloModule);

            exportGeneralizedFixationPolicy<GraphType, false>(aggloModule);
            exportGeneralizedFixationPolicy<GraphType, true>(aggloModule);
        }


   
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
