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

    template<class DATA_BACKEND>
    void exportNodesToBlocksStackedT(py::module & toolsModule) {

        toolsModule.def("nodesToBlocksStacked",
        [](const DATA_BACKEND & segmentation,
           const Blocking<3> & blocking,
           const std::vector<int64_t> & halo,
           const std::vector<int64_t> & skipSlices,
           const int nThreads
        ){
            typedef typename DATA_BACKEND::value_type DataType;
            std::vector<std::vector<DataType>> out;
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

        // export for hdf5
        #ifdef WITH_HDF5
        {
            typedef nifty::hdf5::Hdf5Array<uint32_t> Hdf5Array32;
            typedef nifty::hdf5::Hdf5Array<uint64_t> Hdf5Array64;
            exportNodesToBlocksStackedT<Hdf5Array32>(toolsModule);
            exportNodesToBlocksStackedT<Hdf5Array64>(toolsModule);
        }
        #endif

        // export for z5
        #ifdef WITH_HDF5
        {
            typedef nifty::nz5::DatasetWrapper<uint32_t> Z5Array32;
            typedef nifty::nz5::DatasetWrapper<uint64_t> Z5Array64;
            exportNodesToBlocksStackedT<Z5Array32>(toolsModule);
            exportNodesToBlocksStackedT<Z5Array64>(toolsModule);
        }
        #endif
    }
}
}
