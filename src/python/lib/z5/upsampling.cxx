#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#include "nifty/z5/upsampling.hxx"

namespace py = pybind11;

namespace nifty {
namespace nz5 {

    template<class T>
    void exportUpsamplingT(py::module & m, const std::string & typeName) {
        const std::string name = "nearestUpsampling" + typeName;
        m.def(name.c_str(), [](const std::string & inPath,
                              const std::vector<int> & samplingFactor,
                              const std::string & outPath,
                              const int numberOfThreads) {
            py::gil_scoped_release allowThreads;
            nearestUpsampling<T>(inPath, samplingFactor, outPath, numberOfThreads);
        }, py::arg("inPath"), py::arg("samplingFactor"), py::arg("outPath"), py::arg("numberOfThreads")=-1 );
    }


    void exportIntersectMask(py::module & m) {
        m.def("intersectMasks", [](const std::string & maskAPath,
                                   const std::string & maskBPath,
                                   const std::string & outPath,
                                   const std::vector<size_t> & blockShape,
                                   const int numberOfThreads){
            py::gil_scoped_release allowThreads;
            intersectMasks(maskAPath, maskBPath, outPath, blockShape, numberOfThreads);
        }, py::arg("maskAPath"), py::arg("maskBPath"), py::arg("outPath"),
           py::arg("blockShape"), py::arg("numberOfThreads")=-1);
    }


    void exportUpsampling(py::module & m) {
        exportIntersectMask(m);
        exportUpsamplingT<uint8_t>(m, "Uint8");
    }


}
}
