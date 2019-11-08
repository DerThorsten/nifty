#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"
#include "nifty/distributed/lifted_utils.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportLiftedUtils(py::module & module) {

        // function from z5 data
        module.def("computeLiftedNeighborhoodFromNodeLabels", [](const std::string & graphPath,
                                                                 const std::string & graphKey,
                                                                 const std::string & nodeLabelPath,
                                                                 const std::string & nodeLabelKey,
                                                                 const std::string & outputPath,
                                                                 const std::string & outputKey,
                                                                 const unsigned graphDepth,
                                                                 const int numberOfThreads,
                                                                 const std::string & mode,
                                                                 const uint64_t ignoreLabel){
            py::gil_scoped_release allowThreads;
            computeLiftedNeighborhoodFromNodeLabels(graphPath, graphKey,
                                                    nodeLabelPath, nodeLabelKey,
                                                    outputPath, outputKey,
                                                    graphDepth, numberOfThreads, mode);
        }, py::arg("graphPath"), py::arg("graphKey"),
           py::arg("nodeLabelPath"), py::arg("nodeLabelKey"),
           py::arg("outputPath"), py::arg("outputKey"),
           py::arg("graphDepth"), py::arg("numberOfThreads"), py::arg("mode")="all",
           py::arg("ignoreLabel")=0);

        // function from in memory data
        module.def("liftedNeighborhoodFromNodeLabels", [](const Graph & graph,
                                                          const xt::pytensor<uint64_t, 1> & nodeLabels,
                                                          const int graphDepth,
                                                          const int numberOfThreads,
                                                          const std::string & mode,
                                                          const uint64_t ignoreLabel){
            std::vector<EdgeType> liftedEdges;
            {
                py::gil_scoped_release allowThreads;
                computeLiftedNeighborhoodFromNodeLabels(graph, nodeLabels,
                                                        graphDepth, numberOfThreads,
                                                        liftedEdges, mode, ignoreLabel);
            }
            const int64_t nLifted = liftedEdges.size();
            xt::pytensor<uint64_t, 2> out = xt::zeros<uint64_t>({nLifted, int64_t(2)});

            for(std::size_t liftedId = 0; liftedId < nLifted; ++liftedId) {
                const auto & lifted = liftedEdges[liftedId];
                out(liftedId, 0) = lifted.first;
                out(liftedId, 1) = lifted.second;
            }
            return out;
        }, py::arg("graph"), py::arg("nodeLabels"),
           py::arg("graphDepth"), py::arg("numberOfThreads"),
           py::arg("mode")="all", py::arg("ignoreLabel")=0);

        module.def("liftedEdgesFromNode", [](const std::string & graphPath,
                                             const std::string & graphKey,
                                             const uint64_t srcNode,
                                             const unsigned graphDepth){
            std::vector<EdgeType> tmp;
            {
                py::gil_scoped_release allowThreads;
                const auto graph = Graph(graphPath, graphKey, 1);
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

        }, py::arg("graphPath"), py::arg("graphKey"), py::arg("srcNode"), py::arg("graphDepth"));
    }
}
}
#endif
