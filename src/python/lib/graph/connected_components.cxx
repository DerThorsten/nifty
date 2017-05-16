#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/graph/components.hxx"

#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    template<class GRAPH>
    void exportConnectedComponentsT(py::module & module) {

        // function
        module.def("connectedComponents",
        [](
            const GRAPH & graph,
            nifty::marray::PyView<uint64_t,1> labels
        ){
            nifty::marray::PyView<uint64_t> ccLabels({labels.shape(0)});

            ComponentsUfd<GRAPH> componentsUfd(graph);
            componentsUfd.buildFromLabels(labels);
            for(const auto node : graph.nodes()){
                ccLabels[node] = componentsUfd.componentLabel(node);
            }
            return ccLabels;
        });
    }

    void exportConnectedComponents(py::module & module) {

        {
            typedef UndirectedGraph<> GraphType;
            exportConnectedComponentsT<GraphType>(module);
        }
        
    }

}
}
