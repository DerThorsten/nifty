#pragma once
#ifndef NIFTY_PYTHON_GRAPH_AGGLO_EXPORT_AGGLOMERATIVE_CLUSTERING_HXX
#define NIFTY_PYTHON_GRAPH_AGGLO_EXPORT_AGGLOMERATIVE_CLUSTERING_HXX


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/graph/agglo/agglomerative_clustering.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{
namespace agglo{


    //using namespace py;
    
    template<bool WITH_UCM>
    struct ExportUcmFunctions{
        template<class AGGLO_CLUSTER_TYPE>
        void static exportUcm(py::class_<AGGLO_CLUSTER_TYPE> & aggloCls){

        }
    };

    template<>
    struct ExportUcmFunctions<true>{
        template<class AGGLO_CLUSTER_TYPE>
        void static exportUcm(py::class_<AGGLO_CLUSTER_TYPE> & aggloCls){
            
            aggloCls
                .def("runAndGetDendrogramHeight", [](
                    AGGLO_CLUSTER_TYPE * self
                ){
                    const auto & graph = self->graph();
                    nifty::marray::PyView<uint64_t> dheight( {std::size_t(graph.edgeIdUpperBound()+1)  });
                    {
                        py::gil_scoped_release allowThreads;
                        self->runAndGetDendrogramHeight(dheight);
                    }
                    return dheight;
                }
                )
            ;
        }
    };



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
        auto aggloCls = py::class_<AgglomerativeClusteringType>(aggloModule, aggloClsName.c_str());
           
        aggloCls
            .def("run", [](
                AgglomerativeClusteringType * self
            ){
                {
                    py::gil_scoped_release allowThreds;
                    self->run();
                }
            }
            )

            .def("result", [](
                const AgglomerativeClusteringType * self
            ){
                const auto graph = self->graph();
                nifty::marray::PyView<uint64_t> out({size_t(graph.nodeIdUpperBound()+1)});
                {
                    py::gil_scoped_release allowThreds;
                    self->result(out);
                }
                return out;
            }
            )

            .def("result", [](
                const AgglomerativeClusteringType * self,
                nifty::marray::PyView<uint64_t> out 
            ){
                const auto graph = self->graph();
                {
                    py::gil_scoped_release allowThreds;
                    self->result(out);
                }
                return out;
            },
            py::arg("out")
            )
        ;

        // additional functions which are only enabled if 
        // cluster policies enables ucm
        typedef ExportUcmFunctions<AgglomerativeClusteringType::WithEdgeUfd::value> UcmExporter;
        UcmExporter::exportUcm(aggloCls);


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

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
#endif //NIFTY_PYTHON_GRAPH_AGGLO_EXPORT_AGGLOMERATIVE_CLUSTERING_HXX