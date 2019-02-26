#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/lifted_utils.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportLiftedUtils(py::module & module) {

        module.def("computeLiftedNeighborhoodFromNodeLabels", [](const std::string & graphPath,
                                                                 const std::string & nodeLabelPath,
                                                                 const std::string & outputPath,
                                                                 const unsigned graphDepth,
                                                                 const int numberOfThreads,
                                                                 const std::string & mode){
            py::gil_scoped_release allowThreads;
            computeLiftedNeighborhoodFromNodeLabels(graphPath, nodeLabelPath, outputPath,
                                                    graphDepth, numberOfThreads, mode);
        }, py::arg("graphPath"), py::arg("nodeLabelPath"), py::arg("outputPath"),
           py::arg("graphDepth"), py::arg("numberOfThreads"), py::arg("mode")="all");


        module.def("liftedEdgesFromNode", [](const std::string & graphPath,
                                             const uint64_t srcNode,
                                             const unsigned graphDepth){
            std::vector<EdgeType> tmp;
            {
                py::gil_scoped_release allowThreads;
                const auto graph = Graph(graphPath, 1);
                liftedEdgesFromNode(graph, srcNode, graphDepth, tmp);
            }
            xt::pytensor<uint64_t, 2> out({static_cast<int64_t>(tmp.size()), 2});
            {
                py::gil_scoped_release allowThreads;
                for(std::size_t i = 0; i < tmp.size(); ++i) {
                    out(i, 0) = tmp[i].first;
                    out(i, 1) = tmp[i].second;
                }
            }
            return out;

        }, py::arg("graphPath"), py::arg("srcNode"), py::arg("graphDepth"));
    }
}
}
#endif
