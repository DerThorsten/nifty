#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/agglo/export_agglomerative_clustering.hxx"

#include "nifty/graph/agglo/agglomerative_clustering.hxx"

#include "nifty/graph/agglo/cluster_policies/edge_weighted_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/minimum_node_size_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/lifted_graph_edge_weighted_cluster_policy.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


  
    template<class GRAPH, bool WITH_UCM>
    void exportEdgeWeightedClusterPolicy(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1> PyViewFloat1;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef EdgeWeightedClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("EdgeWeightedClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                .def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                .def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;
        

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat1 & nodeSizes,
                    const uint64_t numberOfNodesStop,
                    const float sizeRegularizer
                ){
                    EdgeWeightedClusterPolicySettings s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.sizeRegularizer = sizeRegularizer;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("numberOfNodesStop") = 1,
                py::arg("sizeRegularizer") = 0.5f
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    template<class GRAPH>
    void exportMinimumNodeSizeClusterPolicy(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1> PyViewFloat1;


        {   
            // name and type of cluster operator
            typedef MinimumNodeSizeClusterPolicy<GraphType> ClusterPolicyType;
            typedef typename ClusterPolicyType::Settings Setting;
            const auto clusterPolicyBaseName = std::string("MinimumNodeSizeClusterPolicy");
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                //.def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;
        

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat1 & nodeSizes,
                    const double minimumNodeSize,
                    const double sizeRegularizer,
                    const double gamma
                ){
                    Setting s;
                    s.minimumNodeSize = minimumNodeSize;
                    s.sizeRegularizer = sizeRegularizer;
                    s.gamma = gamma;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("minimumNodeSize") = 1,
                py::arg("sizeRegularizer") = 0.001,
                py::arg("gamma") = 0.999
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }



    void exportAgglomerativeClustering(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;

            exportEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);
            exportMinimumNodeSizeClusterPolicy<GraphType>(aggloModule);
        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
