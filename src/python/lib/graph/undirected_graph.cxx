#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/graph/simple_graph.hxx"

#include "export_undirected_graph_class_api.hxx"
#include "../converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportUndirectedGraph(py::module & graphModule) {

        typedef UndirectedGraph<> Graph;
        const auto clsName = std::string("UndirectedGraph");
        auto undirectedGraphCls = py::class_<Graph>(graphModule, clsName.c_str());

        undirectedGraphCls
            .def(py::init<const uint64_t,const uint64_t>()
              ,
               py::arg("numberOfNodes"),
               py::arg("reserveEdges") = 0
            )
            .def("insertEdges",
                [](Graph & g, py::array_t<uint64_t> pyArray) {
                    NumpyArray<uint64_t> array(pyArray);
                    NIFTY_CHECK_OP(array.dimension(),==,2,"wrong dimensions");
                    NIFTY_CHECK_OP(array.shape(1),==,2,"wrong shape");
                    for(size_t i=0; i<array.shape(0); ++i){
                        g.insertEdge(array(i,0),array(i,1));
                    }
                }
            )
            .def("__repr__",
                [](const Graph & g) {
                    std::stringstream ss;
                    auto first = true;
                    for(auto edge : g.edges()){
                        if(first){
                            first = false;
                            ss<<g.u(edge)<<"-"<<g.v(edge);
                        }
                        else
                            ss<<"\n"<<g.u(edge)<<"-"<<g.v(edge);
                    }
                    return ss.str();
                }
            )
        ;

        // export the base graph API (others might derive)
        exportUndirectedGraphClassAPI<Graph>(graphModule, undirectedGraphCls,clsName);


    }

}
}
