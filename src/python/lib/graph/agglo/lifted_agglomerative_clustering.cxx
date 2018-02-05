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
#include "nifty/graph/agglo/cluster_policies/detail/merge_rules.hxx"


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


  


    template<class GRAPH, class ACC, bool WITH_UCM>
    void exportLiftedAgglomerativeClusteringPolicyT(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
        typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef LiftedGraphEdgeWeightedClusterPolicy<GraphType, ACC, WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("LiftedGraphEdgeWeightedClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyBaseName2 = clusterPolicyBaseName + ACC::staticName();
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyBaseName2.c_str())
                //.def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;
        

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & mergePrios,
                    const PyViewUInt8_1 & isLocalEdge,
                    const PyViewFloat1 & edgeSizes,
                    const double stopPriority,
                    const typename ClusterPolicyType::AccSettingsType updateRule,
                    const uint64_t numberOfNodesStop
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.stopPriority = stopPriority;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.updateRule = updateRule;

                    auto ptr = new ClusterPolicyType(graph, mergePrios, isLocalEdge, edgeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("mergePrios"),
                py::arg("isMergeEdge"),
                py::arg("edgeSizes"),
                py::arg("stopPriority") = 0.5,
                py::arg("updateRule") = typename ACC::SettingsType(),
                py::arg("numberOfNodesStop") = 1
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName2);
        }
    }


   


    void exportLiftedAgglomerativeClusteringPolicy(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;

            typedef merge_rules::ArithmeticMeanEdgeMap<GraphType, double >  ArithmeticMeanAcc;
            typedef merge_rules::GeneralizedMeanEdgeMap<GraphType, double > GeneralizedMeanAcc;
            typedef merge_rules::SmoothMaxEdgeMap<GraphType, double >       SmoothMaxAcc;
            typedef merge_rules::RankOrderEdgeMap<GraphType, double >       RankOrderAcc;
            typedef merge_rules::MaxEdgeMap<GraphType, double >             MaxAcc;
            typedef merge_rules::MinEdgeMap<GraphType, double >             MinAcc;


            exportLiftedAgglomerativeClusteringPolicyT<GraphType,  ArithmeticMeanAcc,  false>(aggloModule);
            exportLiftedAgglomerativeClusteringPolicyT<GraphType,  GeneralizedMeanAcc, false>(aggloModule);
            exportLiftedAgglomerativeClusteringPolicyT<GraphType,  SmoothMaxAcc,       false>(aggloModule);
            exportLiftedAgglomerativeClusteringPolicyT<GraphType,  RankOrderAcc,       false>(aggloModule);
            exportLiftedAgglomerativeClusteringPolicyT<GraphType,  MaxAcc,             false>(aggloModule);
            exportLiftedAgglomerativeClusteringPolicyT<GraphType,  MinAcc,             false>(aggloModule);

         
        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
