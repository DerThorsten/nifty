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


#include "nifty/graph/agglo/cluster_policies/fixation_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/fixation_cluster_policy2.hxx"
namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


  

    template<class GRAPH, bool WITH_UCM>
    void exportFixationPolicy(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
        typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef FixationClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("FixationClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                .def_property_readonly("mergePrios", &ClusterPolicyType::mergePrios)
                .def_property_readonly("notMergePrios", &ClusterPolicyType::notMergePrios)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;
        

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & mergePrios,
                    const PyViewFloat1 & notMergePrios,
                    const PyViewUInt8_1 & isMergeEdge,
                    const PyViewFloat1 & edgeSizes,
                    const uint64_t numberOfNodesStop
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    auto ptr = new ClusterPolicyType(graph, mergePrios, notMergePrios, isMergeEdge, edgeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("mergePrios"),
                py::arg("notMergePrios"),
                py::arg("isMergeEdge"),
                py::arg("edgeSizes"),
                py::arg("numberOfNodesStop") = 1
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    template<class GRAPH, bool WITH_UCM>
    void exportFixationPolicy2(py::module & aggloModule) {
        
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef nifty::marray::PyView<float, 1>   PyViewFloat1;
        typedef nifty::marray::PyView<uint8_t, 1> PyViewUInt8_1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {   
            // name and type of cluster operator
            typedef FixationClusterPolicy2<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("FixationClusterPolicy2") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                .def_property_readonly("mergePrios", &ClusterPolicyType::mergePrios)
                .def_property_readonly("notMergePrios", &ClusterPolicyType::notMergePrios)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;
        

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & mergePrios,
                    const PyViewFloat1 & notMergePrios,
                    const PyViewUInt8_1 & isLocalEdge,
                    const PyViewFloat1 & edgeSizes,
                    const uint64_t numberOfNodesStop
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    auto ptr = new ClusterPolicyType(graph, mergePrios, notMergePrios, isLocalEdge, edgeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("mergePrios"),
                py::arg("notMergePrios"),
                py::arg("isLocalEdge"),
                py::arg("edgeSizes"),
                py::arg("numberOfNodesStop") = 1
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    void exportFixationAgglomerativeClustering(py::module & aggloModule) {
        {
            typedef PyUndirectedGraph GraphType;

            
            exportFixationPolicy<GraphType, false>(aggloModule);
            exportFixationPolicy<GraphType, true>(aggloModule);

            exportFixationPolicy2<GraphType, false>(aggloModule);
            exportFixationPolicy2<GraphType, true>(aggloModule);
        }

        aggloModule.def(
            "pixelWiseFixation2D",
            [](
                xt::pytensor<float, 3>      mergePrios,
                xt::pytensor<float, 3>      notMergePrios,
                xt::pytensor<int64_t, 2>    offsets,
                xt::pytensor<bool, 1>       isMergeEdgeOffset,
                const bool sparseRepulsiveOnly = true
            ){
                const auto & shape = mergePrios.shape();
                std::array<int,2 > uCoord, vCoord;
                typedef nifty::graph::UndirectedGraph<> GraphType;
                nifty::graph::UndirectedGraph<> g(shape[0] * shape[1]);

                auto vi = [&](const auto & uCoord){
                    return uCoord[0]*shape[1] + uCoord[1];
                };

                for(uCoord[0]=0; uCoord[0]<shape[0]; ++uCoord[0])
                for(uCoord[1]=0; uCoord[1]<shape[1]; ++uCoord[1]){
                    const auto u = vi(uCoord);


                    for(auto offset_index=0; offset_index<shape[2]; ++offset_index){

                        vCoord[0] = uCoord[0] + offsets(offset_index, 0);
                        vCoord[1] = uCoord[1] + offsets(offset_index, 1);

                        if(vCoord[0]>=0 && vCoord[0]<shape[0] && vCoord[1]>=0 && vCoord[1]<shape[1]){
                            const auto v = vi(vCoord);
                            g.insertEdge(u,v);
                        }
                    }
                }

                typedef typename GraphType:: template EdgeMap<float>   FloatEdgeMap;
                typedef typename GraphType:: template EdgeMap<uint8_t> UInt8EdgeMap;
                typedef typename GraphType:: template EdgeMap<uint64_t> UInt64EdgeMap;

                FloatEdgeMap mergePriosMap(g);
                FloatEdgeMap notMergePriosMap(g);
                FloatEdgeMap edgeSizeMap(g,1.0);
                UInt8EdgeMap isMergeEdgeMap(g);
                //UInt64EdgeMap resultMap(g);

                for(uCoord[0]=0; uCoord[0]<shape[0]; ++uCoord[0])
                for(uCoord[1]=0; uCoord[1]<shape[1]; ++uCoord[1]){
                    const auto u = vi(uCoord);


                    for(auto offset_index=0; offset_index<shape[2]; ++offset_index){

                        vCoord[0] = uCoord[0] + offsets(offset_index, 0);
                        vCoord[1] = uCoord[1] + offsets(offset_index, 1);

                        if(vCoord[0]>=0 && vCoord[0]<shape[0] && vCoord[1]>=0 && vCoord[1]<shape[1]){
                            const auto v = vi(vCoord);
                            const auto edge = g.findEdge(u,v);

                            mergePriosMap[edge]    = mergePrios(uCoord[0], uCoord[1], offset_index);
                            notMergePriosMap[edge] = notMergePrios(uCoord[0], uCoord[1], offset_index);
                            isMergeEdgeMap[edge]   = isMergeEdgeOffset(offset_index);
                        }
                    }
                }


                typedef FixationClusterPolicy<GraphType, false> ClusterPolicyType;
                typedef AgglomerativeClustering<ClusterPolicyType> AgglomerativeClusteringType;

                ClusterPolicyType clusterPolicy(g, mergePriosMap, notMergePriosMap, isMergeEdgeMap, edgeSizeMap);
                AgglomerativeClusteringType hcluster(clusterPolicy);
                hcluster.run();


                xt::pytensor<uint64_t, 2> result({
                    shape[0], 
                    shape[1]
                });

                hcluster.result(result);
                return result;

            },
            py::arg("mergePrios"),
            py::arg("notMergePrios"),
            py::arg("offsets"),
            py::arg("isMergeEdgeOffset"),
            py::arg("sparseRepulsiveOnly") = true
        );


        aggloModule.def(
            "pixelWiseFixation3D",
            [](
                xt::pytensor<float, 4  >      mergePrios,
                xt::pytensor<float, 4  >      notMergePrios,
                xt::pytensor<int64_t, 2>    offsets,
                xt::pytensor<bool, 1>       isMergeEdgeOffset
            ){
                const auto & shape = mergePrios.shape();
                std::array<int,3 > uCoord, vCoord;
                typedef nifty::graph::UndirectedGraph<> GraphType;
                GraphType g(shape[0] * shape[1]* shape[2]);

                auto vi = [&](const auto & coord){
                    return coord[0]*shape[1]*shape[2] 
                         + coord[1]*shape[2]   
                         + coord[2];
                };

                for(uCoord[0]=0; uCoord[0]<shape[0]; ++uCoord[0])
                for(uCoord[1]=0; uCoord[1]<shape[1]; ++uCoord[1])
                for(uCoord[2]=0; uCoord[2]<shape[2]; ++uCoord[2])
                {
                    const auto u = vi(uCoord);


                    for(auto offset_index=0; offset_index<shape[3]; ++offset_index){

                        vCoord[0] = uCoord[0] + offsets(offset_index, 0);
                        vCoord[1] = uCoord[1] + offsets(offset_index, 1);
                        vCoord[2] = uCoord[2] + offsets(offset_index, 2);

                        if(vCoord[0]>=0 && vCoord[0]<shape[0] && 
                           vCoord[1]>=0 && vCoord[1]<shape[1] &&
                           vCoord[2]>=0 && vCoord[2]<shape[2]
                        ){
                            if(isMergeEdgeOffset[offset_index]){
                                const auto v = vi(vCoord);
                                g.insertEdge(u,v);
                            }
                            else{
                                const auto mergeP    = mergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                const auto notMergeP = notMergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                if(notMergeP*4.0 > mergeP){
                                    const auto v = vi(vCoord);
                                    g.insertEdge(u,v);
                                }
                            }
                        }
                    }
                }

                typedef typename GraphType:: template EdgeMap<float>   FloatEdgeMap;
                typedef typename GraphType:: template EdgeMap<uint8_t> UInt8EdgeMap;
                typedef typename GraphType:: template EdgeMap<uint64_t> UInt64EdgeMap;

                FloatEdgeMap mergePriosMap(g);
                FloatEdgeMap notMergePriosMap(g);
                FloatEdgeMap edgeSizeMap(g,1.0);
                UInt8EdgeMap isMergeEdgeMap(g);
                UInt64EdgeMap resultMap(g);

                auto c=0;
                for(uCoord[0]=0; uCoord[0]<shape[0]; ++uCoord[0])
                for(uCoord[1]=0; uCoord[1]<shape[1]; ++uCoord[1])
                for(uCoord[2]=0; uCoord[2]<shape[2]; ++uCoord[2])
                {
                    const auto u = vi(uCoord);
                    NIFTY_CHECK_OP(c,==,u,"");

                    for(auto offset_index=0; offset_index<shape[3]; ++offset_index){

                        vCoord[0] = uCoord[0] + offsets(offset_index, 0);
                        vCoord[1] = uCoord[1] + offsets(offset_index, 1);
                        vCoord[2] = uCoord[2] + offsets(offset_index, 2);

                        if(vCoord[0]>=0 && vCoord[0]<shape[0] && 
                           vCoord[1]>=0 && vCoord[1]<shape[1] &&
                           vCoord[2]>=0 && vCoord[2]<shape[2]
                        ){



                            if(isMergeEdgeOffset[offset_index]){
                                const auto v = vi(vCoord);
                                const auto edge = g.findEdge(u,v);

                                mergePriosMap[edge]    = mergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                notMergePriosMap[edge] = notMergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                isMergeEdgeMap[edge]   = isMergeEdgeOffset(offset_index);

                            }
                            else{
                                const auto mergeP    = mergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                const auto notMergeP = notMergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                if(notMergeP*4.0 > mergeP){
                                    const auto v = vi(vCoord);
                                    const auto edge = g.findEdge(u,v);
                                    NIFTY_CHECK_OP(edge,>=,0,"internal error");

                                    mergePriosMap[edge]    = mergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                    notMergePriosMap[edge] = notMergePrios(uCoord[0], uCoord[1], uCoord[2], offset_index);
                                    isMergeEdgeMap[edge]   = isMergeEdgeOffset(offset_index);
                                }
                            }

                        }
                    }
                    ++c;
                }


                typedef FixationClusterPolicy<GraphType, false> ClusterPolicyType;
                typedef AgglomerativeClustering<ClusterPolicyType> AgglomerativeClusteringType;

                ClusterPolicyType clusterPolicy(g, mergePriosMap, notMergePriosMap, isMergeEdgeMap, edgeSizeMap);
                AgglomerativeClusteringType hcluster(clusterPolicy);
                hcluster.run(true,10000);


                xt::pytensor<uint64_t, 3> result({
                    shape[0], 
                    shape[1],
                    shape[2]
                });

                hcluster.result(resultMap);


                c=0;
                for(uCoord[0]=0; uCoord[0]<shape[0]; ++uCoord[0])
                for(uCoord[1]=0; uCoord[1]<shape[1]; ++uCoord[1])
                for(uCoord[2]=0; uCoord[2]<shape[2]; ++uCoord[2])
                {
                    const auto u = vi(uCoord);
                    result(uCoord[0], uCoord[1], uCoord[2]) = resultMap[u];
                    NIFTY_CHECK_OP(c,==,u,"");
                    ++c;
                }


                
                return result;

            },
            py::arg("mergePrios"),
            py::arg("notMergePrios"),
            py::arg("offsets"),
            py::arg("isMergeEdgeOffset")
        );
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
