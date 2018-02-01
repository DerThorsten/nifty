#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/distributed_graph.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportDistributedGraph(py::module & module) {

        py::class_<Graph>(module, "Graph")
            .def(py::init<const std::string &>())
            .def_property_readonly("numberOfNodes", &Graph::numberOfNodes)
            .def_property_readonly("numberOfEdges", &Graph::numberOfEdges)

            .def("findEdge", &Graph::findEdge)   // TODO lift gil

            .def("findEdges", [](const Graph & self, const xt::pytensor<NodeType, 2> uvs){
                typedef xt::pytensor<EdgeIndexType, 1> OutType;
                typedef typename OutType::shape_type OutShape;
                OutShape shape = {uvs.shape()[0]};
                OutType out(shape);
                {
                    py::gil_scoped_release allowThreads;
                    for(size_t i = 0; i < shape[0]; ++i) {
                        out[i] = self.findEdge(uvs[0], uvs[1]);
                    }
                }
                return out;
            })
        ;


    }


}
}
