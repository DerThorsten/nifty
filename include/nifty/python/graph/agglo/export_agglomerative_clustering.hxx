#pragma once


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/graph/agglo/agglomerative_clustering.hxx"

#include <xtensor/xtensor.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor-python/pyarray.hpp>     // Numpy bindings
#include <xtensor-python/pytensor.hpp>     // Numpy bindings


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
                    typedef typename xt::pytensor<uint64_t, 1>::shape_type ShapeType;
                    ShapeType shape = {graph.edgeIdUpperBound() + 1};
                    xt::pytensor<uint64_t, 1> mtimes(shape);
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
                    typedef typename xt::pytensor<uint64_t, 1>::shape_type ShapeType;
                    ShapeType shape = {graph.edgeIdUpperBound() + 1};

                    xt::pytensor<double, 1> dheight(shape);
                    xt::pytensor<uint64_t, 1> mtimes (shape);
                    {
                        py::gil_scoped_release allowThreads;
                        self->runAndGetMergeTimesAndDendrogramHeight(mtimes, dheight,verbose);
                    }
                    return std::make_pair(mtimes, dheight);
                }
                ,
                    py::arg("verbose") = false
                )

                .def("runAndGetDendrogramHeight", [](
                    AGGLO_CLUSTER_TYPE * self, const bool verbose
                ){
                    const auto & graph = self->graph();
                    typedef typename xt::pytensor<double, 1>::shape_type ShapeType;
                    ShapeType shape = {graph.edgeIdUpperBound() + 1};
                    xt::pytensor<double, 1> dheight(shape);
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
                    typedef typename xt::pytensor<double, 1>::shape_type ShapeType;
                    ShapeType shape = {graph.edgeIdUpperBound() + 1};
                    xt::pytensor<double, 1> transformed(shape);
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




    template<class CLUSTER_POLICY, class PY_AGGLO_CLS>
    void exportAgglomerativeClusteringVisitors(
        py::module & aggloModule,
        const std::string & clusterPolicyBaseName,
        PY_AGGLO_CLS & aggloCls
    ){
        typedef CLUSTER_POLICY ClusterPolicyType;
        typedef typename ClusterPolicyType::GraphType GraphType;

        typedef AgglomerativeClustering<ClusterPolicyType> AgglomerativeClusteringType;

        const auto graphName = GraphName<GraphType>::name();
        const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
        //const auto aggloClsName = std::string("AgglomerativeClustering") + clusterPolicyClsName;


        typedef DendrogramAgglomerativeClusteringVisitor<AgglomerativeClusteringType> DendrogramAgglomerativeClusteringVisitorType;

        // dendrogram visitor
        {   
            typedef DendrogramAgglomerativeClusteringVisitorType VisitorType;
            const auto visitorClsName = std::string("DendrogramAgglomerativeClusteringVisitor") 
                + clusterPolicyClsName;

            auto visitorCls = py::class_<VisitorType>(aggloModule, visitorClsName.c_str());

            visitorCls
                .def("dendrogramEncoding",[](
                    VisitorType & visitor
                ){
                    typedef std::is_same<typename GraphType::NodeIdTag,  ContiguousTag> GraphHasContiguousNodeIds;
                    static_assert( GraphHasContiguousNodeIds::value,
                      "dendrogram visitor dendrogramEncoding works only for graphs with contiguous node ids"
                    );
                    //std::cout<<"a\n";

                    const auto & encoding = visitor.dendrogramEncoding();
                    // std::cout<<"encoding.size() "<<encoding.size()<<"\n";
                    xt::pytensor<uint64_t,2> nodes = xt::ones<uint64_t>({std::size_t(encoding.size()),std::size_t(2)});
                    //std::cout<<"b\n";
                    xt::pytensor<double,1>   p = xt::ones<double>({encoding.size()});
                    xt::pytensor<double,1>   s = xt::ones<double>({encoding.size()});
                    //std::cout<<"c\n";
                    uint64_t c = 0;
                    for(const auto e : encoding){
                        nodes(c,0) = std::get<0>(e);
                        nodes(c,1) = std::get<1>(e);
                        p(c) = std::get<2>(e);
                        s(c) = std::get<3>(e);
                        ++c;
                    }
                    // std::cout<<"d\n";
                        
                    return std::make_tuple(nodes, p, s);

                })
            ;

        }




        // addition function in agglo class
        aggloCls.def("dendrogramVisitor",
            [](
                AgglomerativeClusteringType & cls
            ){

                typedef std::is_same<typename GraphType::NodeIdTag,  ContiguousTag> GraphHasContiguousNodeIds;
                static_assert( GraphHasContiguousNodeIds::value,
                  "dendrogram visitor works only for graphs with contiguous node ids"
                );

                return DendrogramAgglomerativeClusteringVisitorType(cls);
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0,1>()
        )
        .def("run", [](
                AgglomerativeClusteringType * self,
                DendrogramAgglomerativeClusteringVisitorType & visitor,
                const bool verbose,
                const int64_t printNth
            ){
                {
                    py::gil_scoped_release allowThreds;
                    self->run(visitor, verbose, printNth);
                }
            }
            ,
                py::arg("visitor"),
                py::arg("verbose") = false,
                py::arg("printNth") = 1
            )
        ;

    }



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

        // the agglomerative cluster policy itself
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
                typedef typename xt::pytensor<uint64_t, 1>::shape_type ShapeType;
                ShapeType shape = {graph.edgeIdUpperBound() + 1};
                xt::pytensor<uint64_t, 1> out(shape);
                {
                    py::gil_scoped_release allowThreds;
                    self->result(out);
                }
                return out;
            }
            )

            .def("result", [](
                const AgglomerativeClusteringType * self,
                xt::pytensor<uint64_t, 1> & out
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

        // visitors
        exportAgglomerativeClusteringVisitors<CLUSTER_POLICY>(aggloModule, clusterPolicyBaseName, aggloCls);



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
    
