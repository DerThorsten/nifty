#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/graph/edge_weighted_watersheds.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    template<class GRAPH>
    void exportEdgeWeightedWatershedsT(py::module & module) {

        // function
        module.def("edgeWeightedWatershedsSegmentation",
            [](
                const GRAPH & graph,
                xt::pytensor<uint64_t, 1> & seeds,
                xt::pytensor<float, 1> & edgeWeights
            ){

                xt::pytensor<uint64_t, 1> labels({uint64_t(seeds.shape()[0])});
                {
                    py::gil_scoped_release allowThreads;
                    edgeWeightedWatershedsSegmentation(graph, edgeWeights, seeds, labels);
                }
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

    void exportEdgeWeightedWatersheds(py::module & module) {

        {
            typedef UndirectedGraph<> GraphType;
            exportEdgeWeightedWatershedsT<GraphType>(module);
        }
        {
            typedef UndirectedGridGraph<2, true> GraphType;
            exportEdgeWeightedWatershedsT<GraphType>(module);
        }
        {
            typedef UndirectedGridGraph<3, true> GraphType;
            exportEdgeWeightedWatershedsT<GraphType>(module);
        }
    }

}
}
