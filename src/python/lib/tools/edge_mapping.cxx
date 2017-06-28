#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
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
                [](MappingType & self, const marray::PyView<EdgeType> uvIds, const std::vector<NodeType> & oldToNewNodes) {
                    self.initializeMapping(uvIds, oldToNewNodes);
                }
            )
            
            .def("mapEdgeValues",
                [](const MappingType & self, const std::vector<float> & edgeValues) {
                    std::vector<float> newEdgeValues;
                    self.mapEdgeValues(edgeValues, newEdgeValues);
                    return newEdgeValues;
                }
            )
            
            .def("getNewUvIds",
                [](const MappingType & self) {
                    size_t shape[] = {self.numberOfNewEdges(), 2};
                    marray::PyView<EdgeType> newUvIds(shape, shape + 2);
                    const auto & newUvIdsInternal = self.getNewUvIds();
                    
                    // this could also be parallelized
                    for(size_t i = 0; i < self.numberOfNewEdges(); ++i) {
                        newUvIds(i, 0) = newUvIdsInternal[i].first;
                        newUvIds(i, 1) = newUvIdsInternal[i].second;
                    }

                    return newUvIds;
                }
            )

            .def("getNewEdgeIds",
                [](const MappingType & self, const std::vector<EdgeType> & edgeIds) {
                    std::vector<EdgeType> newEdgeIds;
                    self.getNewEdgeIds(edgeIds, newEdgeIds);
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
