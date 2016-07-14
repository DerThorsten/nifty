#pragma once
#ifndef NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX
#define NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX

#include "nifty/python/converter.hxx"
#include "export_undirected_graph_class_api.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{




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
        const auto clsName = GraphName<GraphType>::name();

        auto cls  = py::class_< GraphType >(graphModule, clsName.c_str());

        // export the base graph API (others might derive)
        exportUndirectedGraphClassAPI<GraphType>(graphModule, cls, clsName);

        return cls;
    }   
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX
