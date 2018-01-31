#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/distributed/region_graph.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;

namespace nifty {
namespace distributed {


    void exportRegionGraph(py::module & module) {

        typedef nifty::array::StaticArray<int64_t, 3> CoordinateType;
        module.def("computeMergeableRegionGraph", [](
            const std::string & pathToLabels,
            const std::string & keyToLabels,
            const CoordinateType & roiBegin,
            const CoordinateType & roiEnd,
            const std::string & pathToGraph,
            const std::string & keyToGraph
        ) {

            py::gil_scoped_release allowThreads;
            computeMergeableRegionGraph(pathToLabels, keyToLabels,
                                        roiBegin, roiEnd,
                                        pathToGraph, keyToGraph);

        }, py::arg("pathToLabels"), py::arg("keyToLabels"),
           py::arg("roiBegin"), py::arg("roiEnd"),
           py::arg("pathToGraph"), py::arg("keyToGraph"));

    }

}
}
