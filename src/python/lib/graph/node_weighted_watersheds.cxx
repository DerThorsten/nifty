#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"

#include "nifty/graph/node_weighted_watersheds.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{

    template<class GRAPH>
    void exportNodeWeightedWatershedsT(py::module & module) {

        // function
        module.def("nodeWeightedWatershedsSegmentation",
        [](
            const GRAPH & graph,
            xt::pytensor<uint64_t, 1> seeds,
            xt::pytensor<float, 1> nodeWeights
        ){
            xt::pytensor<uint64_t, 1> labels({uint64_t(seeds.shape()[0])});
            nodeWeightedWatershedsSegmentation(graph, nodeWeights, seeds, labels);
            return labels;
        },
            py::arg("graph"),
            py::arg("seeds"),
            py::arg("nodeWeights"),
            "Node weighted watershed on a graph\n\n"
            "Arguments:\n\n"
            "  graph : the input graph\n"
            "   seeds (numpy.ndarray): the seeds\n"
            "   nodeWeights (numpy.ndarray): the node weights\n\n"
            "Returns:\n\n"
            "   numpy.ndarray : the segmentation"
        );
    }

    void exportNodeWeightedWatersheds(py::module & module) {

        {
            typedef UndirectedGraph<> GraphType;
            exportNodeWeightedWatershedsT<GraphType>(module);
        }
        {
            typedef UndirectedGridGraph<2, true> GraphType;
            exportNodeWeightedWatershedsT<GraphType>(module);
        }
        {
            typedef UndirectedGridGraph<3, true> GraphType;
            exportNodeWeightedWatershedsT<GraphType>(module);
        }
    }

}
}
