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
            .def(py::init<const xt::pytensor<EdgeType, 2> &,
                          const xt::pytensor<NodeType, 1> &,
                          const int>(), py::arg("uvIds"), py::arg("nodeLabeling"), py::arg("numberOfThreads")=-1)

            .def("mapEdgeValues",
                [](const MappingType & self, const xt::pytensor<float, 1> & edgeValues, const int numberOfThreads) {
                    typename xt::pytensor<float, 1>::shape_type shape = {static_cast<int64_t>(self.numberOfNewEdges())};
                    xt::pytensor<float, 1 >newEdgeValues(shape);
                    {
                        py::gil_scoped_release allowThreads;
                        self.mapEdgeValues(edgeValues, newEdgeValues, numberOfThreads);
                    }
                    return newEdgeValues;
                }, py::arg("edgeValues"), py::arg("numberOfThreads")=-1
            )

            .def("newUvIds",
                [](const MappingType & self,
                   const int numberOfThreads) {

                    typedef typename xt::pytensor<EdgeType, 2>::shape_type ShapeType;
                    const int64_t nNew = self.numberOfNewEdges();
                    ShapeType shape{nNew, 2L};
                    xt::pytensor<EdgeType, 2> newUvIds(shape);

                    {
                        py::gil_scoped_release allowThreads;
                        const auto & newUvIdsInternal = self.newUvIds();

                        nifty::parallel::ThreadPool threadpool(numberOfThreads);

                        parallel::parallel_foreach(threadpool, nNew, [&](const int tId, const size_t i) {
                            newUvIds(i, 0) = newUvIdsInternal[i].first;
                            newUvIds(i, 1) = newUvIdsInternal[i].second;
                        });
                    }

                    return newUvIds;
                }, py::arg("numberOfThreads")=-1
            )

            .def("getNewEdgeIds",
                [](const MappingType & self, const std::vector<EdgeType> & edgeIds) {
                    std::vector<EdgeType> newEdgeIds;
                    {
                        py::gil_scoped_release allowThreads;
                        self.getNewEdgeIds(edgeIds, newEdgeIds);
                    }
                    return newEdgeIds;
                }, py::arg("edgeIds")
            )
            ;
    }


    void exportEdgeMapping(py::module & toolsModule) {
        exportEdgeMappingT<int64_t, uint64_t>(toolsModule);
    }

}
}
