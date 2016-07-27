#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/graph/undirected_list_graph.hxx"

#include "export_undirected_graph_class_api.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportUndirectedListGraph(py::module & graphModule) {

        typedef UndirectedGraph<> Graph;
        const auto clsName = std::string("UndirectedGraph");
        auto undirectedGraphCls = py::class_<Graph>(graphModule, clsName.c_str());

        undirectedGraphCls
            .def(py::init<const uint64_t,const uint64_t>()
              ,
               //py::arg("numberOfNodes"),
               py::arg("numberOfNodes"),
               py::arg_t<uint64_t>("reserveEdges",0)
            )
            .def("insertEdge",&Graph::insertEdge)
            .def("insertEdges",
                [](Graph & g, nifty::marray::PyView<uint64_t> array) {
                    NIFTY_CHECK_OP(array.dimension(),==,2,"wrong dimensions");
                    NIFTY_CHECK_OP(array.shape(1),==,2,"wrong shape");
                    for(size_t i=0; i<array.shape(0); ++i){
                        g.insertEdge(array(i,0),array(i,1));
                    }
                }
            )
            .def("uvIds",
                [](Graph & g) {
                    nifty::marray::PyView<uint64_t> out({uint64_t(g.numberOfEdges()), uint64_t(2)});
                    for(const auto edge : g.edges()){
                        const auto uv = g.uv(edge); 
                        out(edge,0) = uv.first;
                        out(edge,1) = uv.second;
                    }
                    return out;
                }
            )
        ;

        // export the base graph API (others might derive)
        exportUndirectedListGraphClassAPI<Graph>(graphModule, undirectedGraphCls,clsName);


    }

}
}
