#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/eval_utils.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportEvalUtils(py::module & module) {

        module.def("computeAndSerializeContingencyTable", [](const xt::pytensor<uint64_t, 3> & segA,
                                                             const xt::pytensor<uint64_t, 3> & segB,
                                                             const std::string & path,
                                                             const std::vector<std::size_t> & chunkId,
                                                             const uint64_t ignoreA=0, const uint64_t ignoreB=0) {
            py::gil_scoped_release allowThreads;
            computeAndSerializeContingecyTable(segA, segB, path, chunkId,
                                               ignoreA, ignoreB);
        }, py::arg("segA"), py::arg("segB"),
           py::arg("path"), py::arg("chunkId"),
           py::arg("ignoreA")=0, py::arg("ignoreB")=0);
    }
}
}
#endif
