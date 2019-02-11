#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "nifty/carving/carving.hxx"
#include "nifty/graph/rag/grid_rag.hxx"

namespace py = pybind11;

namespace nifty{
namespace carving{


    template<class GRAPH>
    void exportCarvingT(py::module & module,
                        const std::string & graphName) {

        typedef xt::pytensor<float, 1> WeightsType;
        typedef GRAPH GraphType;
        typedef CarvingSegmenter<GraphType> CarvingType;
        const auto clsName = std::string("CarvingSegmenter") + graphName;
        py::class_<CarvingType>(module, clsName.c_str())
            .def(py::init<const GraphType &, const WeightsType &, bool>(),
                 py::arg("graph"),
                 py::arg("edgeWeights"),
                 py::arg("sortEdges")=true)

            // TODO for some reason pure call by reference does not work
            // and we still need to return the seeds to see a change
            .def("__call__", [](const CarvingType & self,
                                xt::pytensor<uint8_t, 1> & seeds,
                                const double bias,
                                const double noBiasBelow){
                {
                    py::gil_scoped_release allowThreads;
                    self(seeds, bias, noBiasBelow);
                }
                return seeds;
            }, py::arg("seeds"),
               py::arg("bias"),
               py::arg("noBiasBelow"))
        ;

    }


    void exportCarving(py::module & module) {

        typedef xt::pytensor<uint32_t, 2> ExplicitLabels2D;
        typedef graph::GridRag<2, ExplicitLabels2D> Rag2D;
        exportCarvingT<Rag2D>(module, "Rag2D");

        typedef xt::pytensor<uint32_t, 3> ExplicitLabels3D;
        typedef graph::GridRag<3, ExplicitLabels3D> Rag3D;
        exportCarvingT<Rag3D>(module, "Rag3D");
    }

}
}


PYBIND11_MODULE(_carving, module) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();

    module.doc() = "carving submodule of nifty";

    using namespace nifty::carving;
    exportCarving(module);
}
