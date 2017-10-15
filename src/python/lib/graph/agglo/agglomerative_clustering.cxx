#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/agglo/export_agglomerative_clustering.hxx"

#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/agglo/agglomerative_clustering.hxx"
#include "nifty/graph/agglo/cluster_policies/mala_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/edge_weighted_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/node_and_edge_weighted_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/minimum_node_size_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/lifted_graph_edge_weighted_cluster_policy.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


    template<class GRAPH, bool WITH_UCM>
    void exportLiftedGraphEdgeWeightedPolicy(py::module & aggloModule) {
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1> PyViewFloat1;
        typedef nifty::marray::PyView<bool, 1> PyViewBool1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string();

        {
            // name and type of cluster operator
            typedef LiftedGraphEdgeWeightedClusterPolicy<GraphType, PyViewFloat1, PyViewFloat1, PyViewFloat1, PyViewBool1, WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("LiftedGraphEdgeWeightedClusterPolicy") +  withUcmStr;
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
                    const PyViewBool1 & edgeIsLifted,
                    const uint64_t numberOfNodesStop,
                    const float sizeRegularizer
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.sizeRegularizer = sizeRegularizer;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, edgeIsLifted, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("edgeIsLifted"),
                py::arg("numberOfNodesStop") = 1,
                py::arg("sizeRegularizer") = 0.5f
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    template<class GRAPH, bool WITH_UCM>
    void exportMalaClusterPolicy(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1> PyViewFloat1;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef MalaClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("MalaClusterPolicy") +  withUcmStr;
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
                    const float threshold,
                    const uint64_t numberOfNodesStop,
                    const float sizeRegularizer,
                    const bool verbose
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.sizeRegularizer = sizeRegularizer;
                    s.threshold = threshold;
                    s.verbose = verbose;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("threshold") = 0.5,
                py::arg("numberOfNodesStop") = 1,
                py::arg("sizeRegularizer") = 0.5f,
                py::arg("verbose") = false
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }

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

    template<class GRAPH, bool WITH_UCM>
    void exportNodeAndEdgeWeightedClusterPolicy(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1> PyViewFloat1;
        typedef nifty::marray::PyView<float, 2> PyViewFloat2;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef NodeAndEdgeWeightedClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("NodeAndEdgeWeightedClusterPolicy") +  withUcmStr;
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
                    const PyViewFloat2 & nodeFeatures,
                    const PyViewFloat1 & nodeSizes,
                    const float beta,
                    const uint64_t numberOfNodesStop,
                    const float sizeRegularizer
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.sizeRegularizer = sizeRegularizer;
                    s.beta = beta;
                    // create a MultibandArrayViewNodeMap

                    nifty::graph::graph_maps::MultibandArrayViewNodeMap<PyViewFloat2> nodeFeaturesView(nodeFeatures);

                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeFeaturesView, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeFeatures"),
                py::arg("nodeSizes"),
                py::arg("beta") = 0.5f,
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
            typedef typename ClusterPolicyType::SettingsType Setting;
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

            exportMalaClusterPolicy<GraphType, false>(aggloModule);
            exportMalaClusterPolicy<GraphType, true>(aggloModule);

            exportEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportNodeAndEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportNodeAndEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportMinimumNodeSizeClusterPolicy<GraphType>(aggloModule);

            exportLiftedGraphEdgeWeightedPolicy<GraphType, false>(aggloModule);
            exportLiftedGraphEdgeWeightedPolicy<GraphType, true>(aggloModule);
        }

        {
            typedef UndirectedGridGraph<2,true> GraphType;

            exportMalaClusterPolicy<GraphType, false>(aggloModule);
            exportMalaClusterPolicy<GraphType, true>(aggloModule);

            exportEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportNodeAndEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportNodeAndEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportMinimumNodeSizeClusterPolicy<GraphType>(aggloModule);
            
            exportLiftedGraphEdgeWeightedPolicy<GraphType, false>(aggloModule);
            exportLiftedGraphEdgeWeightedPolicy<GraphType, true>(aggloModule);
        }

        {
            typedef UndirectedGridGraph<3,true> GraphType;

            exportMalaClusterPolicy<GraphType, false>(aggloModule);
            exportMalaClusterPolicy<GraphType, true>(aggloModule);

            exportEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportNodeAndEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportNodeAndEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportMinimumNodeSizeClusterPolicy<GraphType>(aggloModule);
            
            exportLiftedGraphEdgeWeightedPolicy<GraphType, false>(aggloModule);
            exportLiftedGraphEdgeWeightedPolicy<GraphType, true>(aggloModule);
        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
