#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/graph/undirected_list_graph.hxx"

#include "export_undirected_graph_class_api.hxx"
#include "nifty/python/converter.hxx"

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
               py::arg_t<uint64_t>("reserveEdges",0)
            )
            .def("insertEdge",&GraphType::insertEdge)
            .def("insertEdges",
                [](GraphType & g, nifty::marray::PyView<uint64_t> array) {
                    NIFTY_CHECK_OP(array.dimension(),==,2,"wrong dimensions");
                    NIFTY_CHECK_OP(array.shape(1),==,2,"wrong shape");
                    for(size_t i=0; i<array.shape(0); ++i){
                        g.insertEdge(array(i,0),array(i,1));
                    }
                }
            )
            .def("serialize",
                [](const GraphType & g) {
                    nifty::marray::PyView<uint64_t> out({g.serializationSize()});
                    auto ptr = &out(0);
                    g.serialize(ptr);
                    return out;
                }
            )
            .def("deserialize",
                [](GraphType & g, nifty::marray::PyView<uint64_t,1> serialization) {

                    auto  startPtr = &serialization(0);
                    auto  lastElement = &serialization(serialization.size()-1);
                    auto d = lastElement - startPtr + 1;

                    NIFTY_CHECK_OP(d,==,serialization.size(), "serialization must be contiguous");



                    g.deserialize(startPtr);
                }
            )
            .def("extractSubgraphFromNodes",
                []( GraphType & g, const std::vector<int64_t> & nodeList) {
                    std::vector<int64_t> innerEdgesVec;
                    std::vector<int64_t> outerEdgesVec;
                    GraphType subgraph;
                    {
                        py::gil_scoped_release allowThreads;
                        subgraph = g.extractSubgraphFromNodes(nodeList, innerEdgesVec, outerEdgesVec);
                    }
                    return std::make_tuple(innerEdgesVec, outerEdgesVec, subgraph);
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
        ;

        graphModule.def("longRangeGridGraph3D",
            [&](
                UndirectedGraph<> & g,
                array::StaticArray<int64_t, 3>      shape,
                nifty::marray::PyView<int64_t,2>   offsets,
                nifty::marray::PyView<float, 4>    affinities
            ){
                g.assign(shape[0]*shape[1]*shape[2]);

                uint64_t u=0;
                for(int p0=0; p0<shape[0]; ++p0)
                for(int p1=0; p1<shape[1]; ++p1)
                for(int p2=0; p2<shape[2]; ++p2){

                    for(int io=0; io<offsets.shape(0); ++io){

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

                nifty::marray::PyView<uint32_t, 1>   offsetsIndex({g.numberOfEdges()});
                nifty::marray::PyView<float>         aff({g.numberOfEdges()});
                

                u=0;
                for(int p0=0; p0<shape[0]; ++p0)
                for(int p1=0; p1<shape[1]; ++p1)
                for(int p2=0; p2<shape[2]; ++p2){

                    for(int io=0; io<offsets.shape(0); ++io){

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
