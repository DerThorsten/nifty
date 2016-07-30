#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

#include "nifty/graph/agglo/agglomerative_clustering.hxx"
#include "nifty/graph/agglo/cluster_policies/edge_weighted_cluster_policy.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


    using namespace py;
    


    template<class CLUSTER_POLICY>
    void exportAgglomerativeClusteringTClusterPolicy(
        py::module & aggloModule,
        const std::string & clusterPolicyBaseName
    ){
        typedef CLUSTER_POLICY ClusterPolicyType;
        typedef typename ClusterPolicyType::GraphType GraphType;

        typedef AgglomerativeClustering<ClusterPolicyType> AgglomerativeClusteringType;

        const auto graphName = GraphName<GraphType>::name();
        const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
        const auto aggloClsName = std::string("AgglomerativeClustering") + clusterPolicyClsName;

        // cls
        py::class_<AgglomerativeClusteringType>(aggloModule, aggloClsName.c_str())
            .def("run",&AgglomerativeClusteringType::run,"run clustering")

            .def("result", [](
                const AgglomerativeClusteringType * self
            ){
                const auto graph = self->graph();
                nifty::marray::PyView<uint64_t> out({size_t(graph.nodeIdUpperBound()+1)});
                self->result(out);
                return out;
            }
            )

            .def("result", [](
                const AgglomerativeClusteringType * self,
                nifty::marray::PyView<uint64_t> out 
            ){
                const auto graph = self->graph();
                self->result(out);
                return out;
            },
            py::arg("out")
            )
        ;


        // factory
        aggloModule.def("agglomerativeClustering",
            [](
                ClusterPolicyType & clusterPolicy
            ){
                auto ptr = new AgglomerativeClusteringType(clusterPolicy);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0,1>(),
            py::arg("clusterPolicy") 
        );


    }




    // export all agglo functionality for a certain graph type
    template<class GRAPH>
    void exportAgglomerativeClusteringTGraph(py::module & aggloModule) {
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();

        typedef nifty::marray::PyView<double, 1> PyViewDoube1;

        {   
            // name and type of cluster operator
            typedef EdgeWeightedClusterPolicy<GraphType,PyViewDoube1,PyViewDoube1,PyViewDoube1> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("EdgeWeightedClusterPolicy");
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
                    PyViewDoube1 nodeSizes
                ){
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes);
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
                py::arg("nodeSizes")
            );


            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    void exportAgglomerativeClustering(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;
            exportAgglomerativeClusteringTGraph<GraphType>(aggloModule);
        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
