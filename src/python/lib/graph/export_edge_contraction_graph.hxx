#pragma once
#ifndef NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX
#define NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX

#include "../converter.hxx"
#include "export_undirected_graph_class_api.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
namespace py = pybind11;


namespace nifty{
namespace graph{


    template<class BASE_GRAPH>
    struct ContractionGraphTypes{
        typedef BASE_GRAPH          BaseGraphType;
        typedef std::set<uint64_t>  SetType;
        typedef FlexibleCallback    CallbackType;
        typedef EdgeContractionGraphWithSets<BaseGraphType, CallbackType, SetType> GraphType;
    };


    template<class BASE_GRAPH>
    py::class_<
        typename ContractionGraphTypes<BASE_GRAPH>::GraphType
    >
    exportEdgeContractionGraph(
        py::module & graphModule,
        const std::string baseGraphName
    ) {

        typedef UndirectedGraph<> Graph;
        const auto clsName = std::string("EdgeContractionGraph") + baseGraphName;
        
        typedef BASE_GRAPH          BaseGraphType;
        typedef std::set<uint64_t>  SetType;
        typedef FlexibleCallback    CallbackType;


        typedef typename ContractionGraphTypes<BASE_GRAPH>::GraphType GraphType;

        auto cls  = py::class_< GraphType >(graphModule, clsName.c_str());

        // export the base graph API (others might derive)
        exportUndirectedGraphClassAPI<GraphType>(graphModule, cls, clsName);

        return cls;
    }   
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_PYTHON_GRAPH_EXPORT_EDGE_CONTRACTION_GRAPH_HXX
