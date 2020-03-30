#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/morphology.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportMorphology(py::module & module) {

        module.def("computeAndSerializeMorphology", [](const xt::pytensor<uint64_t, 3> & labels,
                                                       const std::vector<std::size_t> & coordinateOffset,
                                                       const std::string & outPath,
                                                       const std::string & outKey,
                                                       const std::vector<std::size_t> & chunkId){
            py::gil_scoped_release allowThreads;
            computeAndSerializeMorphology(labels, coordinateOffset,
                                          outPath, outKey, chunkId);
        }, py::arg("labels"), py::arg("coordinateOffset"),
           py::arg("outPath"), py::arg("outKey"), py::arg("chunkId"));


        module.def("mergeAndSerializeMorphology", [](const std::string & inputPath,
                                                     const std::string & inputKey,
                                                     const std::string & outputPath,
                                                     const std::string & outputKey,
                                                     const uint64_t labelBegin,
                                                     const uint64_t labelEnd) {
            mergeAndSerializeMorphology(inputPath, inputKey,
                                        outputPath, outputKey,
                                        labelBegin, labelEnd);
        }, py::arg("inputPath"), py::arg("inputKey"),
           py::arg("outputPath"), py::arg("outputKey"),
           py::arg("labelBegin"), py::arg("labelEnd"));
    }
}
}
#endif
