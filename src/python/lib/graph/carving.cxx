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

        typedef xt::pytensor<float, 1> WeightsType;
        typedef GRAPH GraphType;
        typedef CarvingSegmenter<GraphType> CarvingType;
        const auto clsName = std::string("CarvingSegmenter") + graphName;
        py::class_<CarvingType>(module, clsName.c_str())
            // TODO  we want to make sure graph and edgeWeights stay alive
            .def(py::init<const GraphType &, const WeightsType &, bool>(),
                 py::arg("graph"),
                 py::arg("edgeWeights"),
                 py::arg("sortEdges")=true)

            .def("__call__", [](const CarvingType & self,
                                xt::pytensor<uint8_t, 1> & seeds,
                                const double bias,
                                const double noBiasBelow){
                py::gil_scoped_release allowThreads;
                self(seeds, bias, noBiasBelow);
            }, py::arg("seeds"),
               py::arg("bias"),
               py::arg("noBiasBelow"))
        ;

    }


    void exportCarving(py::module & module) {
        // TODO we actually need to export this for rag !
        typedef UndirectedGraph<> GraphType;
        exportCarvingT<GraphType>(module, "UndirectedGraph");
    }

}
}
