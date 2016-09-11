#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag_features.hxx"


#ifdef WITH_HDF52
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




    void exportGraphAccumulator(py::module & ragModule) {

        typedef UndirectedGraph<> Graph;
        
        // gridRagAccumulateFeatures
        {
            typedef DefaultAccEdgeMap<Graph, double> EdgeMapType;
            typedef DefaultAccNodeMap<Graph, double> NodeMapType;

            // edge map
            {
                
                py::class_<EdgeMapType>(ragModule, "DefaultAccEdgeMapUndirectedGraph")

                    // move implementation to grid_rag_features.hxx instead?
                    .def("getFeatureMatrix",[](EdgeMapType * self){
                            size_t shape[] = {size_t(self->graph().edgeIdUpperBound()+1),size_t(self->numberOfFeatures())};
                            marray::PyView<double> featMat(shape,shape+2);
                            for(size_t e = 0; e < size_t(self->graph().edgeIdUpperBound()+1); e++) {
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
                            size_t shape[] = {size_t(self->graph().nodeIdUpperBound()+1),size_t(self->numberOfFeatures())};
                            marray::PyView<double> featMat(shape, shape+2);
                            for(size_t n = 0; n < self->graph().nodeIdUpperBound()+1; n++) {
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

            // accumulate labels
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);
            
            // export sliced rag (only if we have hdf5 support)
            #ifdef WITH_HDF52
            typedef ChunkedLabelsGridRagSliced<uint32_t> ChunkedLabelsGridRagSliced;
            exportGridRagSlicedAccumulateFeaturesT<ChunkedLabelsGridRagSliced, float, EdgeMapType, NodeMapType>(ragModule);
            exportGridRagSlicedAccumulateLabelsT<ChunkedLabelsGridRagSliced, uint32_t>(ragModule);
            #endif
        }
    }

} // end namespace graph
} // end namespace nifty
    
