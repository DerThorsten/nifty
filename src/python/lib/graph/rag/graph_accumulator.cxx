#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag_features.hxx"

#ifdef WITH_HDF5
#include "vigra/multi_array_chunked_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_features_chunked.hxx"
#endif



namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class RAG,class T,size_t DATA_DIM, class EDGE_MAP, class NODE_MAP>
    void exportGridRagAccumulateFeaturesT(py::module & ragModule){

        ragModule.def("gridRagAccumulateFeatures",
            [](
                const RAG & rag,
                nifty::marray::PyView<T, DATA_DIM> data,
                EDGE_MAP & edgeMap,
                NODE_MAP & nodeMap
            ){  
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateFeatures(rag, data, edgeMap, nodeMap);
                }
            },
            py::arg("graph"),py::arg("data"),py::arg("edgeMap"),py::arg("nodeMap")
        );
    }

    template<class RAG,class T,size_t DATA_DIM>
    void exportGridRagAccumulateLabelsT(py::module & ragModule){

        ragModule.def("gridRagAccumulateLabels",
            [](
                const RAG & rag,
                nifty::marray::PyView<T, DATA_DIM> labels
            ){  
                nifty::marray::PyView<T> nodeLabels({rag.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateLabels(rag, labels, nodeLabels);
                }
                return nodeLabels;

            },
            py::arg("graph"),py::arg("labels")
        );
    }
    
    #ifdef WITH_HDF5
    template<class RAG,class T, class EDGE_MAP, class NODE_MAP>
    void exportGridRagSlicedAccumulateFeaturesT(py::module & ragModule){

        ragModule.def("gridRagSlicedAccumulateFeatures",
            [](
                const RAG & rag,
                nifty::marray::PyView<T, 3> data,
                EDGE_MAP & edgeMap,
                NODE_MAP & nodeMap,
                size_t z0
            ){  
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateFeatures(rag, data, edgeMap, nodeMap, z0);
                }
            },
            py::arg("graph"),py::arg("data"),py::arg("edgeMap"),py::arg("nodeMap"),py::arg("z0")
        );
    }

    template<class RAG,class T>
    void exportGridRagSlicedAccumulateLabelsT(py::module & ragModule){

        ragModule.def("gridRagSlicedAccumulateLabels",
            [](
                const RAG & rag,
                const std::string & labels_file,
                const std::string & labels_key
            ){  
                nifty::marray::PyView<T> nodeLabels({rag.numberOfNodes()});
                {
                    vigra::HDF5File h5_file(labels_file, vigra::HDF5File::ReadOnly);
                    vigra::ChunkedArrayHDF5<3,T> labels(h5_file, labels_key);
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateLabels(rag, labels, nodeLabels);
                }
                return nodeLabels;

            },
            py::arg("graph"),py::arg("labels_file"),py::arg("labels_key")
        );
    }
    #endif



    void exportGraphAccumulator(py::module & ragModule, py::module & graphModule) {

        typedef UndirectedGraph<> Graph;
        
        // gridRagAccumulateFeatures
        {
            typedef DefaultAccEdgeMap<Graph, double> EdgeMapType;
            typedef DefaultAccNodeMap<Graph, double> NodeMapType;

            /* TODO test and export the vigra accumulators
            typedef VigraAccEdgeMap<Graph, double> VigraEdgeMapType;
            typedef VigraAccNodeMap<Graph, double> VigraNodeMapType;

            // vigra edge map
            py::class_<VigraAccEdgeMap>(ragModule, "VigraAccEdgeMapUndirectedGraph")
                .def("getFeatures",
                        []() {
                        
                        })
            */

            // edge map
            {
                
                py::class_<EdgeMapType>(ragModule, "DefaultAccEdgeMapUndirectedGraph")
                    // move implementation to grid_rag_features.hxx instead?
                    .def("getFeatureMatrix",[](EdgeMapType * self){
                            marray::PyView<double> featMat({self->numberOfEdges(),self->numberOfFeatures()});
                            for(size_t e = 0; e < self->numberOfEdges(); e++) {
                                double feats[self->numberOfFeatures()];
                                self->getFeatures(e, feats);
                                // TODO acces row of the view instead 
                                for(size_t f = 0; f < self->numberOfFeatures(); f++) {
                                     featMat(e,f) = feats[f];
                                }
                            }
                            return featMat;
                        })
                ;
                
                ragModule.def("defaultAccEdgeMap", [](const Graph & graph, const double minVal, const double maxVal){

                    EdgeMapType * ptr = nullptr;
                    {
                        py::gil_scoped_release allowThreads;
                        ptr = new EdgeMapType(graph, minVal, maxVal);
                    }
                    return ptr;
                },
                    py::return_value_policy::take_ownership,
                    py::keep_alive<0, 1>(),
                    py::arg("graph"),py::arg("minVal"),py::arg("maxVal")
                );
            }
            // node map
            {
                
                py::class_<NodeMapType>(ragModule, "DefaultAccNodeMapUndirectedGraph")
                    // move implementation to grid_rag_features.hxx instead?
                    .def("getFeatureMatrix",[](NodeMapType * self){
                            marray::PyView<double> featMat({self->numberOfNodes(),self->numberOfFeatures()});
                            for(size_t n = 0; n < self->numberOfNodes(); n++) {
                                double feats[self->numberOfFeatures()];
                                self->getFeatures(n, feats);
                                // TODO acces row of the view instead 
                                for(size_t f = 0; f < self->numberOfFeatures(); f++) {
                                     featMat(n,f) = feats[f];
                                }
                            }
                            return featMat;
                        })
                ;
                
                ragModule.def("defaultAccNodeMap", [](const Graph & graph, const double minVal, const double maxVal){
                    NodeMapType * ptr = nullptr;
                    {
                        py::gil_scoped_release allowThreads;
                        ptr = new NodeMapType(graph, minVal, maxVal);
                    }
                    return ptr;
                },
                    py::return_value_policy::take_ownership,
                    py::keep_alive<0, 1>(),
                    py::arg("graph"),py::arg("minVal"),py::arg("maxVal")
                );
            }

            // accumulate features
            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
            typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;
            exportGridRagAccumulateFeaturesT<ExplicitLabelsGridRag2D, float, 2, EdgeMapType, NodeMapType>(ragModule);
            exportGridRagAccumulateFeaturesT<ExplicitLabelsGridRag3D, float, 3, EdgeMapType, NodeMapType>(ragModule);


            // accumulate labels
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);
            
            // export sliced rag (only if we have hdf5 support)
            #ifdef WITH_HDF5
            typedef ChunkedLabelsGridRagSliced<uint32_t> ChunkedLabelsGridRagSliced;
            exportGridRagSlicedAccumulateFeaturesT<ChunkedLabelsGridRagSliced, float, EdgeMapType, NodeMapType>(ragModule);
            exportGridRagSlicedAccumulateLabelsT<ChunkedLabelsGridRagSliced, uint32_t>(ragModule);
            #endif
        }
    }

} // end namespace graph
} // end namespace nifty
    
