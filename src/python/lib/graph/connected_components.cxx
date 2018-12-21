#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"

#include "nifty/graph/components.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    template<class GRAPH>
    void exportConnectedComponentsT(py::module & module) {


        module.def("connectedComponentsFromNodeLabels",
        [](
            const GRAPH & graph,
            xt::pytensor<uint64_t, 1> nodeLabels,
            const bool dense = true,
            const bool ignoreBackground = false
        ){

            xt::pytensor<uint64_t, 1> ccLabels = xt::zeros<uint64_t>({nodeLabels.shape()[0]});
            ComponentsUfd<GRAPH> componentsUfd(graph);
            componentsUfd.buildFromLabels(nodeLabels);

            for(const auto node : graph.nodes()){
                ccLabels[node] = componentsUfd.componentLabel(node);
            }

            if(dense && ignoreBackground){
                std::unordered_map<uint64_t, uint64_t> mapping;
                for(const auto node: graph.nodes()){

                    const auto nl = nodeLabels[node];
                    const auto ccl = ccLabels[node];
                    if(nl != 0 ){
                        const auto fr = mapping.find(ccl);
                        if(fr==mapping.end()){
                            mapping.emplace(ccl, mapping.size());
                        }
                    }
                }
                for(const auto node: graph.nodes()){
                    const auto nl = nodeLabels[node];
                    const auto ccl = ccLabels[node];
                    if(nl != 0 ){
                        ccLabels[node] = mapping.find(ccl)->second;
                    }
                    else{
                        ccLabels[node] = 0;
                    }
                }
            }
            else if(dense  && !ignoreBackground){
                componentsUfd.denseRelabeling(ccLabels);
            }
            else if(ignoreBackground){
                for(const auto node : graph.nodes()){
                    const auto nl = nodeLabels[node];
                    if(nl == uint64_t(0)){
                        ccLabels[node] = 0;
                    }
                    else{
                        ccLabels[node] += 1;
                    }
                }
            }
            return ccLabels;
        },
            py::arg("graph"),
            py::arg("nodeLabels"),
            py::arg("dense")=true,
            py::arg("ignoreBackground")=false,
            "compute connected component labels of a node labeling\n\n"
            ""
            "All nodes which have zero as nodeLabel will keep a zero"
            ""
            "Arguments:\n\n"
            "  graph : the input graph\n"
            "   nodeLabels (numpy.ndarray): node labeling\n"
            "   dense (bool): should the returned labeling be dense (default {True})\n\n"
            "   ignoreBackground (bool): if true, all input zeros are mapped to zeros (default {False})\n\n"
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
            xt::pytensor<uint64_t, 1> labels
        ){
            self.buildFromLabels(labels);
        })
        .def("buildFromEdgeLabels",[](
            ComponentsType & self,
            xt::pytensor<uint8_t, 1> labels
        ){
            self.buildFromEdgeLabels(labels);
        })
        .def("componentLabels",[](
            ComponentsType & self
        ){
            const auto & g = self.graph();
            const size_t size = g.nodeIdUpperBound()+1;
            xt::pytensor<uint64_t, 1> ccLabels({size});
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
