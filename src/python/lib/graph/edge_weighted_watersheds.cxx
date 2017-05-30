#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"

#include "nifty/graph/edge_weighted_watersheds.hxx"


#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    template<class GRAPH>
    void exportEdgeWeightedWatershedT(py::module & module) {

        // function
        module.def("edgeWeightedWatershedSegmentation",
        [](
            const GRAPH & graph,
            nifty::marray::PyView<uint64_t,1> seeds,
            nifty::marray::PyView<float,1> edgeWeights
        ){
       
            nifty::marray::PyView<uint64_t> labels({seeds.shape(0)});
            
            edgeWeightedWatershedsSegmentation(graph, edgeWeights, seeds, labels);

            return labels;

        },
            py::arg("graph"),
            py::arg("seeds"),
            py::arg("edgeWeights"),
            "Edge weighted watershed on a graph\n\n"
            "Arguments:\n\n"
            "  graph : the input graph\n"
            "   seeds (numpy.ndarray): the seeds\n"
            "   edgeWeights (numpy.ndarray): the edge weights\n\n"
            "Returns:\n\n"
            "   numpy.ndarray : the segmentation"
        );

        


    }

    void exportEdgeWeightedWatershed(py::module & module) {

        {
            typedef UndirectedGraph<> GraphType;
            exportEdgeWeightedWatershedT<GraphType>(module);
        }   
        {
            typedef UndirectedGridGraph<2, true> GraphType;
            exportEdgeWeightedWatershedT<GraphType>(module);
        }
        {
            typedef UndirectedGridGraph<3, true> GraphType;
            exportEdgeWeightedWatershedT<GraphType>(module);
        }
    }

}
}
