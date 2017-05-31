#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"

#include "nifty/graph/components.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    template<class GRAPH>
    void exportConnectedComponentsT(py::module & module) {

        // function
        module.def("connectedComponentsFromNodeLabels",
        [](
            const GRAPH & graph,
            nifty::marray::PyView<uint64_t,1> nodeLabels,
            const bool dense = true
        ){
       
            nifty::marray::PyView<uint64_t> ccLabels({nodeLabels.shape(0)});
            ComponentsUfd<GRAPH> componentsUfd(graph);
            componentsUfd.buildFromLabels(nodeLabels);
            for(const auto node : graph.nodes()){
                ccLabels[node] = componentsUfd.componentLabel(node);
            }
            if(dense){
                componentsUfd.denseRelabeling(ccLabels);
            }
            return ccLabels;

        },
            py::arg("graph"),
            py::arg("nodeLabels"),
            py::arg("dense")=true,
            "compute connected component labels of a node labeling\n\n"
            "Arguments:\n\n"
            "  graph : the input graph\n"
            "   nodeLabels (numpy.ndarray): node labeling\n"
            "   dense (bool): should the returned labeling be dense (default {True})\n\n"
            "Returns:\n\n"
            "   numpy.ndarray : connected components labels"
        );

        typedef GRAPH GraphType;
        typedef ComponentsUfd<GraphType> ComponentsType;
        const auto clsName = std::string("Components") + GraphName<GraphType>::name();
        auto componentsPyCls = py::class_<ComponentsType>(module, clsName.c_str());

        
        module.def("components",
            [](
                const GraphType & graph
            ){
                return new ComponentsType(graph);
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("graph")
        );
        
        componentsPyCls
        .def("build",[](ComponentsType & self){
            self.build();
        })
        .def("buildFromNodeLabels",[](
            ComponentsType & self,
            nifty::marray::PyView<uint64_t,1> labels
        ){
            self.buildFromLabels(labels);
        })
        .def("componentLabels",[](
            ComponentsType & self
        ){
            const auto & g = self.graph();
            const size_t size = g.nodeIdUpperBound()+1;
            nifty::marray::PyView<uint64_t> ccLabels({size});
            for(const auto node : g.nodes()){
                ccLabels[node] = self.componentLabel(node);
            }
            return ccLabels;
        })
        ;


    }

    void exportConnectedComponents(py::module & module) {

        {
            typedef UndirectedGraph<> GraphType;
            exportConnectedComponentsT<GraphType>(module);
        }   
        {
            typedef UndirectedGridGraph<2, true> GraphType;
            exportConnectedComponentsT<GraphType>(module);
        }
        {
            typedef UndirectedGridGraph<3, true> GraphType;
            exportConnectedComponentsT<GraphType>(module);
        }
    }

}
}
