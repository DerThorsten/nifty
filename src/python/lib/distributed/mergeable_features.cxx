#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/distributed/mergeable_features.hxx"

namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportMergeableFeatures(py::module & module) {

        module.def("extractBlockFeaturesFromBoundaryMaps", [](const std::string & groupPath,
                                                              const std::string & blockPrefix,
                                                              const std::string & dataPath,
                                                              const std::string & dataKey,
                                                              const std::string & labelPath,
                                                              const std::string & labelKey,
                                                              const std::vector<size_t> & blockIds,
                                                              float dataMin, float dataMax) {
            py::gil_scoped_release allowThreads;
            

        }, py::arg("groupPath"), py::arg("blockPrefix"),
           py::arg("dataPath"), py::arg("dataKey"),
           py::arg("labelPath"), py::arg("labelKey"),
           py::arg("blockIds"), py::arg("dataMin")=0., py::arg("dataMax")=1.);

    }


}
}
