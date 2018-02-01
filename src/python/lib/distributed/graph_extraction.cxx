#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/distributed/graph_extraction.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;

namespace nifty {
namespace distributed {


    void exportGraphExtraction(py::module & module) {

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


        module.def("mergeSubgraphs", [](
            const std::string & pathToGraph,
            const std::string & blockGroup,
            const std::string & blockPrefix,
            const std::vector<size_t> & blockIds,
            const std::string & outKey,
            const int numberOfThreads
        ) {
            py::gil_scoped_release allowThreads;
            mergeSubgraphs(pathToGraph, blockGroup,
                           blockPrefix, blockIds,
                           outKey, numberOfThreads);
        }, py::arg("pathToGraph"), py::arg("blockGroup"),
           py::arg("blockPrefix"), py::arg("blockIds"),
           py::arg("outKey"), py::arg("numberOfThreads")=1);


        module.def("mapEdgeIds", [](
            const std::string & pathToGraph,
            const std::string & graphGroup,
            const std::string & blockGroup,
            const std::string & blockPrefix,
            const std::vector<size_t> & blockIds,
            const int numberOfThreads
        ) {
            py::gil_scoped_release allowThreads;
            mapEdgeIds(pathToGraph, graphGroup, blockGroup, blockPrefix, blockIds, numberOfThreads);
        }, py::arg("pathToGraph"), py::arg("graphGroup"), py::arg("blockGroup"),
           py::arg("blockPrefix"), py::arg("blockIds"), py::arg("numberOfThreads")=1);


        module.def("mapEdgeIdsForAllBlocks", [](
            const std::string & pathToGraph,
            const std::string & graphGroup,
            const std::string & blockGroup,
            const std::string & blockPrefix,
            const size_t numberOfBlocks,
            const int numberOfThreads
        ) {
            py::gil_scoped_release allowThreads;
            mapEdgeIds(pathToGraph, graphGroup, blockGroup, blockPrefix, numberOfBlocks, numberOfThreads);
        }, py::arg("pathToGraph"), py::arg("graphGroup"), py::arg("blockGroup"),
           py::arg("blockPrefix"), py::arg("numberOfBlocks"), py::arg("numberOfThreads")=1);

    }

}
}
