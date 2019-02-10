#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "xtensor-python/pytensor.hpp"
#include "nifty/graph/carving.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace py = pybind11;

namespace nifty{
namespace graph{


    template<class GRAPH>
    void exportCarvingT(py::module & module,
                        const std::string & graphName) {

        typedef GRAPH GraphType;
        typedef CarvingSegmenter<GraphType> CarvingType;
        const auto clsName = std::string("CarvingSegmenter") + graphName;
        py::class_<CarvingType>(module, clsName.c_str())
            // TODO this could be done more elegantly ....
            // constructors with and without serialization
            .def(py::init<const GraphType &, const xt::pytensor<float, 1> &, bool>(),
                 py::arg("graph"),
                 py::arg("edgeWeights").noconvert(),
                 py::arg("fromSerialization")=false)

            .def(py::init<const GraphType &, const xt::pytensor<std::size_t, 1> &, bool>(),
                 py::arg("graph"),
                 py::arg("edgeWeights").noconvert(),
                 py::arg("fromSerialization")=true)

            .def("__call__", [](const CarvingType & self,
                                const xt::pytensor<uint8_t, 1> & seeds){
                xt::pytensor<uint8_t, 1> nodeLabels = xt::zeros<uint8_t>({self.nNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    self(seeds, nodeLabels);
                }
                return nodeLabels;
            })
        ;

    }


    void exportCarving(py::module & module) {
        // TODO we actually need to export this for rag !
        typedef UndirectedGraph<> GraphType;
        exportCarvingT<GraphType>(module, "UndirectedGraph");
    }

}
}
