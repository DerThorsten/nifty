#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"
#include "nifty/tools/edge_mapping.hxx"

namespace py = pybind11;

namespace nifty{
namespace tools{

    template<class EDGE_TYPE, class NODE_TYPE>
    void exportEdgeMappingT(py::module & toolsModule) {

        typedef EDGE_TYPE EdgeType;
        typedef NODE_TYPE NodeType;

        typedef EdgeMapping<EdgeType, NodeType> MappingType;
        py::class_<MappingType>(toolsModule, "EdgeMapping")
            .def(py::init<size_t>())

            .def("initializeMapping",
                [](MappingType & self, const xt::pytensor<EdgeType, 2> & uvIds, const std::vector<NodeType> & oldToNewNodes) {
                    {
                        py::gil_scoped_release allowThreads;
                        self.initializeMapping(uvIds, oldToNewNodes);
                    }
                }
            )

            .def("mapEdgeValues",
                [](const MappingType & self, const std::vector<float> & edgeValues) {
                    std::vector<float> newEdgeValues;
                    {
                        py::gil_scoped_release allowThreads;
                        self.mapEdgeValues(edgeValues, newEdgeValues);
                    }
                    return newEdgeValues;
                }
            )

            .def("getNewUvIds",
                [](const MappingType & self) {

                    typedef typename xt::pytensor<EdgeType, 2>::shape_type ShapeType;
                    ShapeType shape{(int64_t)self.numberOfNewEdges(), 2L};
                    xt::pytensor<EdgeType, 2> newUvIds(shape);

                    {
                        py::gil_scoped_release allowThreads;
                        const auto & newUvIdsInternal = self.getNewUvIds();

                        // this could also be parallelized
                        for(size_t i = 0; i < self.numberOfNewEdges(); ++i) {
                            newUvIds(i, 0) = newUvIdsInternal[i].first;
                            newUvIds(i, 1) = newUvIdsInternal[i].second;
                        }
                    }

                    return newUvIds;
                }
            )

            .def("getNewEdgeIds",
                [](const MappingType & self, const std::vector<EdgeType> & edgeIds) {
                    std::vector<EdgeType> newEdgeIds;
                    {
                        py::gil_scoped_release allowThreads;
                        self.getNewEdgeIds(edgeIds, newEdgeIds);
                    }
                    return newEdgeIds;
                }
            )
            ;
    }


    void exportEdgeMapping(py::module & toolsModule) {
        exportEdgeMappingT<int64_t, int64_t>(toolsModule);
    }

}
}
