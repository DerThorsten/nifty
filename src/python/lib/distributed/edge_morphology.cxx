#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include "nifty/distributed/edge_morphology.hxx"

namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportEdgeMorphology(py::module & module) {
        module.def("find1DEdges", [](const std::string & blockPrefix,
                                     const std::string & labelPath,
                                     const std::string & labelKey,
                                     const std::size_t numberOfEdges,
                                     const std::vector<std::size_t> & blockIds) {
            xt::pytensor<uint8_t, 1> out = xt::zeros<uint8_t>({numberOfEdges});
            {
                py::gil_scoped_release allowThreads;
                find1DEdges(blockPrefix, labelPath, labelKey, blockIds, out);
            }
            return out;

        }, py::arg("blockPrefix"),
           py::arg("labelPath"), py::arg("labelKey"),
           py::arg("numberOfEdges"), py::arg("blockIds"));
    }

}
}
#endif
