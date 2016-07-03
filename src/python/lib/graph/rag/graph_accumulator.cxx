#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../converter.hxx"

#include "nifty/graph/rag/grid_rag_features.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class RAG,class T,unsigned int DATA_DIM, class EDGE_MAP, class NODE_MAP>
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


    void exportGraphAccumulator(py::module & ragModule, py::module & graphModule) {

        typedef UndirectedGraph<> Graph;
        
        // gridRagAccumulateFeatures
        {
            typedef DefaultAccEdgeMap<Graph, double> EdgeMapType;
            typedef DefaultAccNodeMap<Graph, double> NodeMapType;

            // edge map
            {
                
                py::class_<EdgeMapType>(ragModule, "DefaultAccEdgeMapUndirectedGraph")
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

            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag3D;

            exportGridRagAccumulateFeaturesT<ExplicitLabelsGridRag2D, float, 2, EdgeMapType, NodeMapType>(ragModule);
        }
    }

} // end namespace graph
} // end namespace nifty
    
