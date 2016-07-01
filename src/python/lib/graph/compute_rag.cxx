#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>


#include "export_undirected_graph_class_api.hxx"
#include "../converter.hxx"

#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/rag.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportComputeRag(py::module & graphModule) {

        typedef UndirectedGraph<> Graph;
        

        graphModule.def("computeRag",[](
            Graph & graph,  
            py::array_t<uint64_t> pyLables
        ){
            NumpyArray<uint64_t> labels(pyLables);
            computeRag(graph, labels);
        });

    }

}
}
