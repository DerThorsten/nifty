#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/distributed_graph.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportDistributedGraph(py::module & module) {

        py::class_<Graph>(module, "Graph")
            .def(py::init<const std::string &>())
            .def_property_readonly("numberOfNodes", &Graph::numberOfNodes)
            .def_property_readonly("numberOfEdges", &Graph::numberOfEdges)

            .def("findEdge", &Graph::findEdge)   // TODO lift gil

            .def("findEdges", [](const Graph & self, const xt::pytensor<NodeType, 2> uvs){
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

            .def("extractSubgraphFromNodes", [](const Graph & self, const xt::pytensor<uint64_t, 1> & nodes) {
                //
                std::vector<EdgeIndexType> innerEdgesVec, outerEdgesVec;
                std::vector<EdgeType> uvIdsVec;
                {
                    py::gil_scoped_release allowThreads;
                    self.extractSubgraphFromNodes(nodes, uvIdsVec, innerEdgesVec, outerEdgesVec);
                }

                //
                typedef typename xt::pytensor<EdgeIndexType, 1>::shape_type ShapeType;
                ShapeType innerShape = {static_cast<int64_t>(innerEdgesVec.size())};
                xt::pytensor<EdgeIndexType, 1> innerEdges(innerShape);

                ShapeType outerShape = {static_cast<int64_t>(outerEdgesVec.size())};
                xt::pytensor<EdgeIndexType, 1> outerEdges(outerShape);

                typedef typename xt::pytensor<NodeType, 2>::shape_type UvShapeType;
                UvShapeType uvShape = {static_cast<int64_t>(uvIdsVec.size()), 2L};
                xt::pytensor<NodeType, 2> uvIds(uvShape);

                {
                    py::gil_scoped_release allowThreads;
                    for(size_t i = 0; i < innerEdgesVec.size(); ++i) {
                        innerEdges(i) = innerEdgesVec[i];
                    }
                    for(size_t i = 0; i < outerEdgesVec.size(); ++i) {
                        outerEdges(i) = outerEdgesVec[i];
                    }
                    for(size_t i = 0; i < uvIdsVec.size(); ++i) {
                        uvIds(i, 0) = uvIdsVec[i].first;
                        uvIds(i, 1) = uvIdsVec[i].second;
                    }

                }
                return std::make_tuple(innerEdges, outerEdges, uvIds);


            }, py::arg("nodes"))

            ;


    }


}
}
