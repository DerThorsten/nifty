#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/ground_truth/post_process_carving_ground_truth.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{

    template<size_t DIM>
    void exportPostProcessCarvingNeuroGroundTruthT(py::module & groundTruthModule){

        groundTruthModule.def("postProcessCarvingNeuroGroundTruth",
        [](
            marray::PyView<float, DIM , false>   growMap,
            marray::PyView<uint32_t, DIM, false> groundTruth,
            int shrinkSizeObjects = 2,
            int shrinkSizeBg = 3,
            const uint16_t numberOfQueues = 256,
            const int numberOfThreads = -1,
            const int verbose = 1
        ){
            marray::PyView<uint32_t, DIM> gtOut(groundTruth.shapeBegin(), groundTruth.shapeEnd());
            
            postProcessCarvingNeuroGroundTruth<DIM>(growMap, groundTruth, gtOut,
                shrinkSizeObjects, shrinkSizeBg, numberOfQueues, numberOfThreads, verbose);

            return gtOut;
        },
            py::arg("growMap"),
            py::arg("groundTruth"),
            py::arg("shrinkSizeObjects") = 2,
            py::arg("shrinkSizeBg") = 3,
            py::arg("numberOfQueues") = 256,
            py::arg("mumberOfThreads") = -1,
            py::arg("verbose") = 1
        );
    }


    void exportPostProcessCarvingNeuroGroundTruth(py::module & groundTruthModule){

        exportPostProcessCarvingNeuroGroundTruthT<2>(groundTruthModule);
        exportPostProcessCarvingNeuroGroundTruthT<3>(groundTruthModule);
    }
}
}