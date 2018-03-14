#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include "nifty/mws/mws.hxx"


namespace py = pybind11;

namespace nifty {
namespace mws {


    void exportMws(py::module & module) {

        module.def("computeMwsClustering", [](const uint32_t number_of_labels,
                                              const xt::pytensor<uint32_t, 2> & uvs,
                                              const xt::pytensor<uint32_t, 2> & mutex_uvs,
                                              const xt::pytensor<float, 1> & weights,
                                              const xt::pytensor<float, 1> & mutex_weights) {

            xt::pytensor<uint32_t, 1> node_labeling = xt::zeros<uint32_t>({(int64_t) number_of_labels});
            // TODO lift gil
            computeMwsClustering(number_of_labels, uvs, mutex_uvs, weights, mutex_weights, node_labeling);
            return node_labeling;
        }, py::arg("number_of_labels"),
           py::arg("uvs"), py::arg("mutex_uvs"),
           py::arg("weights"), py::arg("mutex_weights"));
    }
}
}
