#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/graph/undirected_list_graph.hxx"

#include "export_undirected_graph_class_api.hxx"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

namespace nifty{
namespace graph{


    void exportUndirectedListGraph(py::module & graphModule) {

        typedef UndirectedGraph<> GraphType;
        const auto clsName = std::string("UndirectedGraph");
        auto undirectedGraphCls = py::class_<GraphType>(graphModule, clsName.c_str());

        undirectedGraphCls
            .def(py::init<const uint64_t,const uint64_t>(),
               py::arg_t<uint64_t>("numberOfNodes",0),
               py::arg_t<uint64_t>("reserveEdges",0),
               py::call_guard<py::gil_scoped_release>()
            )
            .def("insertEdge", &GraphType::insertEdge)
            .def("insertEdges",
                [](GraphType & g, const xt::pytensor<uint64_t, 2> & array) {
                    NIFTY_CHECK_OP(array.shape()[1],==,2,"wrong shape");
                    for(size_t i=0; i<array.shape()[0]; ++i){
                        g.insertEdge(array(i,0), array(i,1));
                    }
                }, py::arg("array"), py::call_guard<py::gil_scoped_release>()
            )
            .def("serialize",
                [](const GraphType & g) {
                    typename xt::pytensor<uint64_t, 1>::shape_type shape = {static_cast<int64_t>(g.serializationSize())};
                    xt::pytensor<uint64_t, 1> out(shape);
                    auto ptr = &out(0);
                    g.serialize(ptr);
                    return out;
                }
            )
            .def("deserialize",
                [](GraphType & g, xt::pytensor<uint64_t, 1> & serialization) {

                    auto startPtr = &serialization(0);
                    auto lastElement = &serialization(serialization.size()-1);
                    auto d = lastElement - startPtr + 1;

                    NIFTY_CHECK_OP(d,==,serialization.size(),
                                   "serialization must be contiguous");
                    g.deserialize(startPtr);
                }
            )
            .def("extractSubgraphFromNodes",
                 []( GraphType & g, const xt::pytensor<uint64_t, 1> & nodeList) {
                    std::vector<int64_t> innerEdgesVec;
                    std::vector<int64_t> outerEdgesVec;
                    std::vector<std::pair<uint64_t, uint64_t>> subUvsVec;
                    {
                        py::gil_scoped_release allowThreads;
                        g.extractSubgraphFromNodes(nodeList,
                                                   innerEdgesVec,
                                                   outerEdgesVec,
                                                   subUvsVec);
                    }
                    xt::pytensor<int64_t, 1> innerEdges = xt::zeros<int64_t>({static_cast<int64_t>(innerEdgesVec.size())});
                    xt::pytensor<int64_t, 1> outerEdges = xt::zeros<int64_t>({static_cast<int64_t>(outerEdgesVec.size())});
                    xt::pytensor<uint64_t, 2> subUvs({static_cast<int64_t>(subUvsVec.size()), 2});
                    {
                        py::gil_scoped_release allowThreads;
                        for(size_t i = 0; i < innerEdgesVec.size(); ++i) {
                            innerEdges(i) = innerEdgesVec[i];
                        }
                        for(size_t i = 0; i < outerEdgesVec.size(); ++i) {
                            outerEdges(i) = outerEdgesVec[i];
                        }
                        for(size_t i = 0; i < subUvsVec.size(); ++i) {
                            subUvs(i, 0) = subUvsVec[i].first;
                            subUvs(i, 1) = subUvsVec[i].second;
                        }

                    }
                    return std::make_tuple(innerEdges, outerEdges, subUvs);
                }
            )
            .def("edgesFromNodeList",
                [](GraphType & g, const std::vector<int64_t> & nodeList) {

                    std::vector<int64_t> edges;
                    {
                        py::gil_scoped_release allowThreads;
                        g.edgesFromNodeList(nodeList, edges);
                    }
                    return edges;
                }
            )
            .def("shrinkToFit",&GraphType::shrinkToFit)
        ;

        graphModule.def("longRangeGridGraph3D",
            [&](
                UndirectedGraph<> & g,
                array::StaticArray<int64_t, 3>      shape,
                const xt::pytensor<int64_t, 2> &   offsets,
                const xt::pytensor<float, 4> &   affinities
            ){
                g.assign(shape[0]*shape[1]*shape[2]);

                uint64_t u=0;
                for(int p0=0; p0<shape[0]; ++p0)
                for(int p1=0; p1<shape[1]; ++p1)
                for(int p2=0; p2<shape[2]; ++p2){

                    for(int io=0; io<offsets.shape()[0]; ++io){

                        const int q0 = p0 + offsets(io, 0);
                        const int q1 = p1 + offsets(io, 1);
                        const int q2 = p2 + offsets(io, 2);

                        if(q0>=0 && q0<shape[0] &&
                           q1>=0 && q1<shape[1] &&
                           q2>=0 && q2<shape[2]){

                            const auto v = q0*shape[1]*shape[2] + q1*shape[2] + q2;
                            const auto e = g.insertEdge(u, v);
                        }
                    }
                    ++u;
                }

                typedef typename xt::pytensor<uint32_t, 1>::shape_type TensorShapeType;
                TensorShapeType tensorShape = {static_cast<int64_t>(g.numberOfEdges())};
                xt::pytensor<uint32_t, 1>   offsetsIndex(tensorShape);
                xt::pytensor<float, 1>      aff({tensorShape});

                u=0;
                for(int p0=0; p0<shape[0]; ++p0)
                for(int p1=0; p1<shape[1]; ++p1)
                for(int p2=0; p2<shape[2]; ++p2){

                    for(int io=0; io<offsets.shape()[0]; ++io){

                        const int q0 = p0 + offsets(io, 0);
                        const int q1 = p1 + offsets(io, 1);
                        const int q2 = p2 + offsets(io, 2);

                        if(q0>=0 && q0<shape[0] &&
                           q1>=0 && q1<shape[1] &&
                           q2>=0 && q2<shape[2]){

                            const auto v = q0*shape[1]*shape[2] + q1*shape[2] + q2;
                            const auto e = g.findEdge(u, v);
                            offsetsIndex(e) = io;
                            aff(e) = affinities(io, p0, p1, p2);
                        }
                    }
                    ++u;
                }
                return std::make_pair(aff, offsetsIndex);
            }
        );

        // export the base graph API (others might derive)
        exportUndirectedGraphClassAPI<GraphType>(graphModule, undirectedGraphCls,clsName);
    }
}
}
