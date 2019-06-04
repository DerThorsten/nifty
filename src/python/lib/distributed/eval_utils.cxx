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


        module.def("computeEvalPrimitives", [](const std::string & inputPath,
                                               const std::size_t nPoints,
                                               const std::size_t nLabelsA,
                                               const std::size_t nLabelsB,
                                               const int nThreads) {
            std::map<std::string, double> ret;
            {
                py::gil_scoped_release allowThreads;
                computeEvalPrimitives(inputPath, nPoints, nLabelsA, nLabelsB, ret, nThreads);
            }
            return ret;
        }, py::arg("inputPath"), py::arg("nPoints"), py::arg("nLabelsA"), py::arg("nLabelsB"), py::arg("nThreads")=1);
    }
}
}
#endif
