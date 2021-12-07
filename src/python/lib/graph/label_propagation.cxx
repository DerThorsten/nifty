#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/graph/label_propagation.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{

    template<class GRAPH>
    void exportLabelPropagationT(py::module & module) {


        module.def("runLabelPropagation_impl",
        [](
            const GRAPH & graph,
            xt::pytensor<uint64_t, 1> nodeLabels,
            const xt::pytensor<double, 1> signedWeights,
            const xt::pytensor<uint8_t, 1> localEdges,
            const uint64_t nb_iter=1,
            const int64_t size_constr=-1,
            const int64_t nbThreads=-1

        ){
            {
                py::gil_scoped_release allowThreads;
                nifty::graph::runLabelPropagation(graph,nodeLabels,signedWeights,localEdges,nb_iter,size_constr,nbThreads);
            }
        },
            py::arg("graph"),
            py::arg("nodeLabels"),
            py::arg("signedWeights"),
            py::arg("localEdges"),
            py::arg("nb_iter")=1,
            py::arg("size_constr")=-1,
            py::arg("nbThreads")=-1
        );


    }

    void exportLabelPropagation(py::module & module) {

        {
            typedef UndirectedGraph<> GraphType;
            exportLabelPropagationT<GraphType>(module);
        }
        // {
        //     typedef UndirectedGridGraph<2, true> GraphType;
        //     exportLabelPropagationT<GraphType>(module);
        // }
        // {
        //     typedef UndirectedGridGraph<3, true> GraphType;
        //     exportLabelPropagationT<GraphType>(module);
        // }
    }

}
}
