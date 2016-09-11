#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/agglo/export_agglomerative_clustering.hxx"

#include "nifty/graph/agglo/agglomerative_clustering.hxx"
#include "nifty/graph/agglo/cluster_policies/edge_weighted_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/lifted_graph_edge_weighted_cluster_policy.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


  
    // export all agglo functionality for a certain graph type
    template<class GRAPH, bool WITH_UCM>
    void exportAgglomerativeClusteringTGraph(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<double, 1, false> PyViewDoube1;

        const std::string withUcmStr =  WITH_UCM ? std::string() : std::string("WithUcm");

        {   
            // name and type of cluster operator
            typedef EdgeWeightedClusterPolicy<GraphType,PyViewDoube1,PyViewDoube1,PyViewDoube1,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("EdgeWeightedClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            auto clusterPolicyPyCls = py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str());
        

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    PyViewDoube1 edgeIndicators,
                    PyViewDoube1 edgeSizes,
                    PyViewDoube1 nodeSizes,
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
                py::keep_alive<0,2>(), // edgeIndicators
                py::keep_alive<0,3>(), // edgeSizes
                py::keep_alive<0,4>(), // nodeSizes
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


    void exportAgglomerativeClustering(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;
            exportAgglomerativeClusteringTGraph<GraphType, false>(aggloModule);
            exportAgglomerativeClusteringTGraph<GraphType, true>(aggloModule);
        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
