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
                [](const MappingType & self,
                    const xt::pytensor<float, 1> & edgeValues,
                    const std::string & accumulation,
                    const int numberOfThreads) {

                    //
                    typename xt::pytensor<float, 1>::shape_type shape = {static_cast<int64_t>(self.numberOfNewEdges())};

                    float initialValue;
                    if(accumulation == "sum" || accumulation == "mean") {
                        initialValue = 0;
                    } else if(accumulation == "max") {
                        initialValue = std::numeric_limits<float>::min(); // lowest float value
                    } else if(accumulation == "min") {
                        initialValue = std::numeric_limits<float>::max(); // highest float value
                    } else {
                        throw std::runtime_error("Invalid accumulation function: " + accumulation);
                    }

                    xt::pytensor<float, 1> newEdgeValues = xt::pytensor<float, 1>(shape, initialValue);
                    {
                        py::gil_scoped_release allowThreads;
                        if(accumulation == "sum" || accumulation == "mean") {
                            self.mapEdgeValues(edgeValues,
                                               newEdgeValues,
                                               [](const float * acc, float * val){*val += *acc;},
                                               initialValue,
                                               numberOfThreads);
                        } else if(accumulation == "max") {
                            self.mapEdgeValues(edgeValues,
                                               newEdgeValues,
                                               [](const float * acc, float * val){*val = std::max(*val, *acc);},
                                               initialValue,
                                               numberOfThreads);
                        } else if(accumulation == "min") {
                            self.mapEdgeValues(edgeValues,
                                               newEdgeValues,
                                               [](const float * acc, float * val){*val = std::min(*val, *acc);},
                                               initialValue,
                                               numberOfThreads);
                        }

                        // for mean accumulation, we need to divide by the edge counts
                        if(accumulation == "mean") {
                            auto & edgeCounts = self.newEdgeCounts();
                            for(std::size_t ii = 0; ii < edgeCounts.size(); ++ii) {
                                newEdgeValues(ii) /= edgeCounts[ii];
                            }
                        }
                    }
                    return newEdgeValues;
                }, py::arg("edgeValues"), py::arg("accumulation"), py::arg("numberOfThreads")=-1
            )

            .def("newUvIds", [](const MappingType & self,
                                const int numberOfThreads) {

                    typedef typename xt::pytensor<NodeType, 2>::shape_type ShapeType;
                    const int64_t nNew = self.numberOfNewEdges();
                    ShapeType shape{nNew, 2L};
                    xt::pytensor<NodeType, 2> newUvIds(shape);

                    {
                        py::gil_scoped_release allowThreads;
                        const auto & newUvIdsInternal = self.newUvIds();

                        nifty::parallel::ThreadPool threadpool(numberOfThreads);

                        parallel::parallel_foreach(threadpool, nNew, [&](const int tId, const std::size_t i) {
                            newUvIds(i, 0) = newUvIdsInternal[i].first;
                            newUvIds(i, 1) = newUvIdsInternal[i].second;
                        });
                    }

                    return newUvIds;
                }, py::arg("numberOfThreads")=-1
            )

            .def("edgeMapping", [](const MappingType & self,
                                   const int numberOfThreads) {
                    typedef typename xt::pytensor<EdgeType, 1>::shape_type ShapeType;
                    ShapeType shape = {static_cast<int64_t>(self.numberOfEdges())};
                    xt::pytensor<EdgeType, 1> edgeMapping(shape);

                    {
                        py::gil_scoped_release allowThreads;
                        const auto & edgeMappingInternal = self.edgeMapping();
                        nifty::parallel::ThreadPool threadpool(numberOfThreads);

                        parallel::parallel_foreach(threadpool, edgeMappingInternal.size(), [&](const int tId, const std::size_t i) {
                            edgeMapping(i) = edgeMappingInternal[i];
                        });
                    }

                    return edgeMapping;
            }, py::arg("numberOfThreads")=-1)

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
