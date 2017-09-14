#pragma once
#ifndef NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX
#define NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX

#include <pybind11/functional.h> 

#include "nifty/python/converter.hxx"
#include "export_undirected_graph_class_api.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportEdgeContractionGraphCallback(
        py::module & graphModule
    ){
        typedef FlexibleCallback Callback;
        py::class_< Callback >(graphModule, "EdgeContractionGraphCallbackImpl")
            .def(py::init<>())
            .def_readwrite("contractEdgeCallback",&Callback::contractEdgeCallback)
            .def_readwrite("mergeEdgesCallback",&Callback::mergeEdgesCallback)
            .def_readwrite("mergeNodesCallback",&Callback::mergeNodesCallback)
            .def_readwrite("contractEdgeDoneCallback",&Callback::contractEdgeDoneCallback)
        ;
    }


    template<class BASE_GRAPH>
    py::class_<
        PyContractionGraph<BASE_GRAPH>
    >
    exportEdgeContractionGraph(
        py::module & graphModule,
        const std::string baseGraphName
    ) {



        typedef BASE_GRAPH BaseGraphType;
        typedef PyContractionGraph<BaseGraphType> GraphType;
        typedef FlexibleCallback Callback;

        const auto clsName = GraphName<GraphType>::name();

        auto cls  = py::class_< GraphType >(graphModule, clsName.c_str());
        cls
            .def(py::init<const BaseGraphType & ,  Callback & >(),
                py::keep_alive<1,2>(),
                py::keep_alive<1,3>()
            )

            // modifications
            .def("contractEdge", &GraphType::contractEdge)
            .def("reset",&GraphType::reset)


            // queries
            .def("findRepresentativeNode", [](GraphType & g, const uint64_t node){
                return g.findRepresentativeNode(node);
            })
            .def("findRepresentativeEdge", [](GraphType & g, const uint64_t edge){
                return g.findRepresentativeEdge(edge);
            })
            .def("nodeOfDeadEdge", [](const GraphType & g, const uint64_t deadEdge){
                return g.nodeOfDeadEdge(deadEdge);
            })
            .def_property_readonly("baseGraph", &GraphType::baseGraph)



        ;


        // export the base graph API (others might derive)
        exportUndirectedGraphClassAPI<GraphType>(graphModule, cls, clsName);



        return cls;
    }



} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX
