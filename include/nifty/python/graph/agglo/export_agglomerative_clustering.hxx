#pragma once


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
            
            typedef typename AGGLO_CLUSTER_TYPE::GraphType GraphType;
            typedef typename GraphType:: template EdgeMap<double> EdgeMapFloat64;

            aggloCls

                .def("runAndGetMergeTimes", [](
                    AGGLO_CLUSTER_TYPE * self, const bool verbose
                ){
                    const auto & graph = self->graph();
                    nifty::marray::PyView<uint64_t> mtimes ( {std::size_t(graph.edgeIdUpperBound()+1)  });
                    {
                        py::gil_scoped_release allowThreads;
                        self->runAndGetMergeTimes(mtimes, verbose);
                    }
                }
                ,
                    py::arg("verbose") = false
                )

                .def("runAndGetMergeTimesAndDendrogramHeight", [](
                    AGGLO_CLUSTER_TYPE * self, const bool verbose
                ){
                    const auto & graph = self->graph();
                    nifty::marray::PyView<double>   dheight( {std::size_t(graph.edgeIdUpperBound()+1)  });
                    nifty::marray::PyView<uint64_t> mtimes ( {std::size_t(graph.edgeIdUpperBound()+1)  });
                    {
                        py::gil_scoped_release allowThreads;
                        self->runAndGetMergeTimesAndDendrogramHeight(mtimes, dheight,verbose);
                    }
                    return std::pair<
                        nifty::marray::PyView<uint64_t>,
                        nifty::marray::PyView<double> 
                    >(mtimes, dheight);
                }
                ,
                    py::arg("verbose") = false
                )

                .def("runAndGetDendrogramHeight", [](
                    AGGLO_CLUSTER_TYPE * self, const bool verbose
                ){
                    const auto & graph = self->graph();
                    nifty::marray::PyView<double> dheight( {std::size_t(graph.edgeIdUpperBound()+1)  });
                    {
                        py::gil_scoped_release allowThreads;
                        self->runAndGetDendrogramHeight(dheight,verbose);
                    }
                    return dheight;
                }
                ,
                    py::arg("verbose") = false
                )

                .def("ucmTransform", [](
                    AGGLO_CLUSTER_TYPE * self,
                    const EdgeMapFloat64 & edgeValues
                ){
                    const auto & graph = self->graph();
                    nifty::marray::PyView<double> transformed( {std::size_t(graph.edgeIdUpperBound()+1)  });
                    {
                        py::gil_scoped_release allowThreads;
                        self->ucmTransform(edgeValues, transformed);
                    }
                    return transformed;
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
                AgglomerativeClusteringType * self,
                const bool verbose,
                const int64_t printNth
            ){
                {
                    py::gil_scoped_release allowThreds;
                    self->run(verbose, printNth);
                }
            }
            ,
                py::arg("verbose") = false,
                py::arg("printNth") = 1
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
    
