#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/python/converter.hxx"
#include "nifty/distributed/graph_extraction.hxx"
#include "nifty/distributed/graph_tools.hxx"

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
            const std::string & keyToGraph,
            const bool ignoreLabel,
            const bool increaseRoi
        ) {

            py::gil_scoped_release allowThreads;
            computeMergeableRegionGraph(pathToLabels, keyToLabels,
                                        roiBegin, roiEnd,
                                        pathToGraph, keyToGraph,
                                        ignoreLabel, increaseRoi);

        }, py::arg("pathToLabels"), py::arg("keyToLabels"),
           py::arg("roiBegin"), py::arg("roiEnd"),
           py::arg("pathToGraph"), py::arg("keyToGraph"),
           py::arg("ignoreLabel")=false,
           py::arg("increaseRoi")=false);


        module.def("mergeSubgraphs", [](
            const std::string & graphPath,
            const std::string & blockPrefix,
            const std::vector<std::size_t> & blockIds,
            const std::string & outKey,
            const int numberOfThreads
        ) {
            py::gil_scoped_release allowThreads;
            mergeSubgraphs(graphPath, blockPrefix, blockIds,
                           outKey, numberOfThreads);
        }, py::arg("graphPath"), py::arg("blockPrefix"), py::arg("blockIds"),
           py::arg("outKey"), py::arg("numberOfThreads")=1);


        module.def("mapEdgeIds", [](
            const std::string & pathToGraph,
            const std::string & graphGroup,
            const std::string & blockPrefix,
            const std::vector<std::size_t> & blockIds,
            const int numberOfThreads
        ) {
            py::gil_scoped_release allowThreads;
            mapEdgeIds(pathToGraph, graphGroup, blockPrefix, blockIds, numberOfThreads);
        }, py::arg("pathToGraph"), py::arg("graphGroup"),
           py::arg("blockPrefix"), py::arg("blockIds"), py::arg("numberOfThreads")=1);


        module.def("mapEdgeIdsForAllBlocks", [](
            const std::string & pathToGraph,
            const std::string & graphGroup,
            const std::string & blockPrefix,
            const std::size_t numberOfBlocks,
            const int numberOfThreads
        ) {
            py::gil_scoped_release allowThreads;
            mapEdgeIds(pathToGraph, graphGroup, blockPrefix, numberOfBlocks, numberOfThreads);
        }, py::arg("pathToGraph"), py::arg("graphGroup"),
           py::arg("blockPrefix"), py::arg("numberOfBlocks"), py::arg("numberOfThreads")=1);


        module.def("loadAsUndirectedGraphWithRelabeling", [](const std::string & pathToGraph,
                                                             const std::string & graphKey) {
            nifty::graph::UndirectedGraph<> g;
            std::unordered_map<NodeType, NodeType> relabeling;
            {
                py::gil_scoped_release allowThreads;
                loadNiftyGraph(pathToGraph, graphKey, g, relabeling, true);
            }
            return std::make_pair(g, relabeling);
        }, py::arg("pathToGraph"), py::arg("graphKey"));


        module.def("loadAsUndirectedGraph", [](const std::string & pathToGraph,
                                               const std::string & graphKey) {
            nifty::graph::UndirectedGraph<> g;
            std::unordered_map<NodeType, NodeType> relabeling;
            {
                py::gil_scoped_release allowThreads;
                loadNiftyGraph(pathToGraph, graphKey, g, relabeling, false);
            }
            return g;
        }, py::arg("pathToGraph"), py::arg("graphKey"));


        module.def("loadNodes", [](const std::string & pathToGraph, const std::string & graphKey) {
            int64_t nNodes;
            {
                py::gil_scoped_release allowThreads;
                z5::filesystem::handle::File graphFile(pathToGraph);
                z5::filesystem::handle::Group graph(graphFile, graphKey);
                nlohmann::json j;
                z5::readAttributes(graph, j);
                nNodes = j["numberOfNodes"];
            }
            xt::pytensor<NodeType, 1> nodes = xt::zeros<NodeType>({nNodes});
            {
                py::gil_scoped_release allowThreads;
                loadNodesToArray(pathToGraph, graphKey, nodes);
            }
            return nodes;
        }, py::arg("pathToGraph"), py::arg("graphKey"));


        module.def("nodeLabelingToPixels", [](const std::string & labelsPath,
                                              const std::string & labelsKey,
                                              const std::string & outPath,
                                              const std::string & outKey,
                                              const xt::pytensor<NodeType, 1> & nodeLabeling,
                                              const std::vector<std::size_t> & blockIds,
                                              const std::vector<std::size_t> & blockShape) {
            py::gil_scoped_release allowThreads;
            nodeLabelingToPixels(labelsPath, labelsKey, outPath, outKey, nodeLabeling, blockIds, blockShape);
        }, py::arg("labelsPath"), py::arg("labelsKey"),
           py::arg("outPath"), py::arg("outKey"),
           py::arg("nodeLabeling"), py::arg("blockIds"),
           py::arg("blockShape"));


        /*
        module.def("extractSubgraphFromNodes", [](const xt::pytensor<uint64_t, 1> & nodes,
                                                  const std::string & graphBlockPrefix,
                                                  const CoordType & shape,
                                                  const CoordType & blockShape,
                                                  const std::size_t startBlockId) {
            //
            std::vector<EdgeIndexType> innerEdgesVec, outerEdgesVec;
            std::vector<EdgeType> uvIdsVec;
            {
                py::gil_scoped_release allowThreads;
                extractSubgraphFromNodes(nodes, graphBlockPrefix,
                                         shape, blockShape, startBlockId,
                                         uvIdsVec, innerEdgesVec, outerEdgesVec);
            }

            //
            typedef typename xt::pytensor<EdgeIndexType, 1>::shape_type ShapeType;
            ShapeType innerShape = {static_cast<int64_t>(innerEdgesVec.size())};
            xt::pytensor<EdgeIndexType, 1> innerEdges = xt::zeros<EdgeIndexType>(innerShape);

            ShapeType outerShape = {static_cast<int64_t>(outerEdgesVec.size())};
            xt::pytensor<EdgeIndexType, 1> outerEdges = xt::zeros<EdgeIndexType>(outerShape);

            typedef typename xt::pytensor<NodeType, 2>::shape_type UvShapeType;
            UvShapeType uvShape = {static_cast<int64_t>(uvIdsVec.size()), 2L};
            xt::pytensor<NodeType, 2> uvIds(uvShape);

            {
                py::gil_scoped_release allowThreads;
                for(std::size_t i = 0; i < innerEdgesVec.size(); ++i) {
                    innerEdges(i) = innerEdgesVec[i];
                }
                for(std::size_t i = 0; i < outerEdgesVec.size(); ++i) {
                    outerEdges(i) = outerEdgesVec[i];
                }
                for(std::size_t i = 0; i < uvIdsVec.size(); ++i) {
                    uvIds(i, 0) = uvIdsVec[i].first;
                    uvIds(i, 1) = uvIdsVec[i].second;
                }

            }
            return std::make_tuple(innerEdges, outerEdges, uvIds);


        }, py::arg("nodes"), py::arg("graphBlockPrefix"),
           py::arg("shape"), py::arg("blockShape"), py::arg("startBlockId"));
        */


        module.def("serializeMergedGraph", [](const std::string & graphPath,
                                              const std::string & graphBlockPrefix,
                                              const CoordType & shape,
                                              const CoordType & blockShape,
                                              const CoordType & newBlockShape,
                                              const std::vector<std::size_t> & newBlockIds,
                                              const xt::pytensor<NodeType, 1> & nodeLabeling,
                                              const xt::pytensor<EdgeIndexType, 1> & edgeLabeling,
                                              const std::string & outPath,
                                              const std::string & graphOutPrefix,
                                              const int numberOfThreads,
                                              const bool serializeEdges) {
            py::gil_scoped_release allowThreads;
            serializeMergedGraph(graphPath, graphBlockPrefix, shape,
                                 blockShape, newBlockShape, newBlockIds,
                                 nodeLabeling, edgeLabeling,
                                 outPath, graphOutPrefix,
                                 numberOfThreads, serializeEdges);
        }, py::arg("graphPath"),
           py::arg("graphBlockPrefix"),
           py::arg("shape"),
           py::arg("blockShape"),
           py::arg("newBlockShape"),
           py::arg("newBlockIds"),
           py::arg("nodeLabeling"),
           py::arg("edgeLabeling"),
           py::arg("outPath"),
           py::arg("graphOutPrefix"),
           py::arg("numberOfThreads")=-1,
           py::arg("serializeEdges")=true);


        module.def("connectedComponents", [](const Graph & graph,
                                             const xt::pytensor<bool, 1> & edgeLabels){
            xt::pytensor<NodeType, 1> labels = xt::zeros<NodeType>({getNumberOfNodes(graph)});
            {
                py::gil_scoped_release allowThreads;
                connectedComponents(graph, edgeLabels, labels);
            }
            return labels;
        }, py::arg("graph"), py::arg("edgeLabels"));


        module.def("connectedComponentsFromNodes", [](const Graph & graph,
                                                      const xt::pytensor<NodeType, 1> & nodeLabels,
                                                      const bool ignoreLabel){
            xt::pytensor<NodeType, 1> out = xt::zeros<NodeType>({getNumberOfNodes(graph)});
            {
                py::gil_scoped_release allowThreads;
                connectedComponentsFromNodes(graph, nodeLabels, ignoreLabel, out);
            }
            return out;
        }, py::arg("graph"), py::arg("nodeLabels"), py::arg("ignoreLabel"));
    }
}
}
#endif
