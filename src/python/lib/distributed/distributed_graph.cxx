#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/distributed_graph.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportDistributedGraph(py::module & module) {

        py::class_<Graph>(module, "Graph")
            .def(py::init<const std::string &, const int>(),
                 py::arg("path"), py::arg("numberOfThreads")=1)
            .def_property_readonly("numberOfNodes", &Graph::numberOfNodes)
            .def_property_readonly("numberOfEdges", &Graph::numberOfEdges)

            .def("findEdge", &Graph::findEdge)   // TODO lift gil

            .def("findEdges", [](const Graph & self,
                                 const xt::pytensor<NodeType, 2> uvs){
                typedef xt::pytensor<EdgeIndexType, 1> OutType;
                typedef typename OutType::shape_type OutShape;
                OutShape shape = {uvs.shape()[0]};
                OutType out(shape);
                {
                    py::gil_scoped_release allowThreads;
                    for(size_t i = 0; i < shape[0]; ++i) {
                        out[i] = self.findEdge(uvs[0], uvs[1]);
                    }
                }
                return out;
            })

            .def("uvIds", [](const Graph & self){
                typedef xt::pytensor<NodeType, 2> OutType;
                typedef typename OutType::shape_type OutShape;
                OutShape shape = {static_cast<int64_t>(self.numberOfEdges()), 2L};
                OutType out(shape);
                {
                    py::gil_scoped_release allowThreads;
                    const auto & edges = self.edges();
                    size_t edgeId = 0;
                    for(const auto edge : edges) {
                        out(edgeId, 0) = edge.first;
                        out(edgeId, 1) = edge.second;
                        ++edgeId;
                    }
                }
                return out;
            })

            .def("nodes", [](const Graph & self){
                typedef xt::pytensor<NodeType, 1> OutType;
                xt::pytensor<NodeType, 1> nodes = xt::zeros<NodeType>({self.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    std::set<NodeType> nodesTmp;
                    self.nodes(nodesTmp);

                    size_t nodeId = 0;
                    for(const auto node : nodesTmp) {
                        nodes(nodeId) = node;
                        ++nodeId;
                    }
                }
                return nodes;
            })

            .def("extractSubgraphFromNodes", [](const Graph & self,
                                                const xt::pytensor<uint64_t, 1> & nodes,
                                                const bool allowInvalidNodes) {
                //
                std::vector<EdgeIndexType> innerEdgesVec, outerEdgesVec;
                {
                    py::gil_scoped_release allowThreads;
                    self.extractSubgraphFromNodes(nodes, allowInvalidNodes,
                                                  innerEdgesVec, outerEdgesVec);
                }

                //
                typedef typename xt::pytensor<EdgeIndexType, 1>::shape_type ShapeType;
                ShapeType innerShape = {static_cast<int64_t>(innerEdgesVec.size())};
                xt::pytensor<EdgeIndexType, 1> innerEdges(innerShape);

                ShapeType outerShape = {static_cast<int64_t>(outerEdgesVec.size())};
                xt::pytensor<EdgeIndexType, 1> outerEdges(outerShape);

                {
                    py::gil_scoped_release allowThreads;
                    for(size_t i = 0; i < innerEdgesVec.size(); ++i) {
                        innerEdges(i) = innerEdgesVec[i];
                    }
                    for(size_t i = 0; i < outerEdgesVec.size(); ++i) {
                        outerEdges(i) = outerEdgesVec[i];
                    }

                }
                return std::make_pair(innerEdges, outerEdges);
            }, py::arg("nodes"), py::arg("allowInvalidNodes")=false)

            ;
    }


}
}
#endif
