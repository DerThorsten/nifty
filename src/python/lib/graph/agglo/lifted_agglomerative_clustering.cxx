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
#include "nifty/graph/agglo/cluster_policies/lifted_edge_weighted_cluster_policy2.hxx"



namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


  

    template<class GRAPH, bool WITH_UCM>
    void exportLiftedAgglomerativeClusteringPolicyT(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
        typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef LiftedGraphEdgeWeightedClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("LiftedGraphEdgeWeightedClusterPolicy") +  withUcmStr;
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
                    const PyViewUInt8_1 & isLiftedEdge,
                    const PyViewFloat1 & nodeSizes
                ){
                    typename ClusterPolicyType::SettingsType s;
                    //s.numberOfNodesStop = numberOfNodesStop;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, isLiftedEdge, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("isLiftedEdge"),
                py::arg("nodeSizes")
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


   


    void exportLiftedAgglomerativeClusteringPolicy(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;

            
            exportLiftedAgglomerativeClusteringPolicyT<GraphType, false>(aggloModule);
            exportLiftedAgglomerativeClusteringPolicyT<GraphType, true>(aggloModule);

         
        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
