#include <pybind11/pybind11.h>

#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/tools/nodes_to_blocks.hxx"

namespace py = pybind11;

namespace nifty{
namespace tools{

    template<class T>
    void exportNodesToBlocksStackedT(py::module & toolsModule) {
        
        toolsModule.def("nodesToBlocksStacked",
        [](const nifty::hdf5::Hdf5Array<T> & segmentation,
            const Blocking<3> & blocking,
            const std::vector<int64_t> & halo,
            const std::vector<int64_t> & skipSlices,
            const int nThreads
        ){
            std::vector<std::vector<T>> out; 
            {
                py::gil_scoped_release allowThreads;
                nodesToBlocksStacked(segmentation, blocking, halo, skipSlices, out, nThreads);
            }
            return out;
        },
        py::arg("segmentation"),
        py::arg("blocking"),
        py::arg("halo"),
        py::arg("skipSlices"),
        py::arg("nThreads")=-1
        );
    }
    
    void exportNodesToBlocks(py::module & toolsModule) {
        exportNodesToBlocksStackedT<uint32_t>(toolsModule);
        //exportNodesToBlocksStackedT<uint64_t>(toolsModule);
    }
}
}
