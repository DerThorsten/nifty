#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/lifted_utils.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportLiftedUtils(py::module & module) {

        module.def("computeLiftedNeighborhoodFromNodeLabels", [](const std::string & graphPath,
                                                                 const std::string & nodeLabelPath,
                                                                 const std::string & outputPath,
                                                                 const unsigned graphDepth,
                                                                 const int numberOfThreads){
            py::gil_scoped_release allowThreads;
            computeLiftedNeighborhoodFromNodeLabels(graphPath, nodeLabelPath, outputPath,
                                                    graphDepth, numberOfThreads);
        }, py::arg("graphPath"), py::arg("nodeLabelPath"), py::arg("outputPath"),
           py::arg("graphDepth"), py::arg("numberOfThreads"));
    }
}
}
#endif
