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


#include "nifty/graph/agglo/cluster_policies/generalized_long_range_cluster_policy.hxx"
namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


  

    template<class GRAPH, bool WITH_UCM>
    void exportGeneralizedLongRangeClusterPolicyT(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
        typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
        typedef nifty::marray::PyView<uint64_t, 1> PyViewUInt64_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef GeneralizedLongRangeClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("GeneralizedLongRangeClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
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
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat1 & nodeSizes,
                    const PyViewUInt8_1 & isLocalEdge,
                    const PyViewUInt64_1 & seeds,
                    const uint64_t stopNodeNumber,
                    const bool useSeeds,
                    const double minSize,
                    const double sizeRegularizer
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.stopNodeNumber = stopNodeNumber;
                    s.useSeeds = useSeeds;
                    s.minSize = minSize;
                    s.sizeRegularizer = sizeRegularizer;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators,edgeSizes,nodeSizes, isLocalEdge, seeds, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("isLocalEdge"),
                py::arg("seeds"),
                py::arg("stopNodeNumber") = 1,
                py::arg("useSeeds") = false,
                py::arg("minSize") = 0.0,
                py::arg("sizeRegularizer") = 0.0
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }




    void exportGeneralizedLongRangeClusterPolicy(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;  
            exportGeneralizedLongRangeClusterPolicyT<GraphType, false>(aggloModule);
            exportGeneralizedLongRangeClusterPolicyT<GraphType, true>(aggloModule);

        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
