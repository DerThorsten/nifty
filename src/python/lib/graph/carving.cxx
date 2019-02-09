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
            .def(py::init<const GraphType &, const xt::pytensor<float, 1> &>())
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


    void exportCarvingT(py::module & module) {
        // TODO we actually need to export this for rag !
        typedef UndirectedGraph<> GraphType;
        exportCarvingT<GraphType>(module, "UndirectedGraph");
    }

}
}
