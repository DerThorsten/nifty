#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/graph/undirected_grid_graph.hxx"

#include "export_undirected_graph_class_api.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    template<size_t DIM>
    void exportUndirectedGridGraphT(py::module & module) {

        typedef UndirectedGridGraph<DIM,true> GraphType;
        const auto clsName = std::string("UndirectedGridGraph") +
                        std::to_string(DIM) + std::string("DSimpleNh");

        auto graphCls = py::class_<GraphType>(module, clsName.c_str());

        graphCls
            .def(py::init<const typename GraphType::ShapeType>(),
               py::arg("shape")
            )
            .def("nodeToCoordinate",[](
                const GraphType & g,
                const uint64_t node
            ){
                return g.nodeToCoordinate(node);
            })
            .def("coordianteToNode",[](
                const GraphType & g,
                const typename GraphType::CoordinateType & coord
            ){
                return g.coordianteToNode(coord);
            })
            //.def("uvIds",
            //    [](GraphType & g) {
            //        nifty::marray::PyView<uint64_t> out({uint64_t(g.numberOfEdges()), uint64_t(2)});
            //        for(const auto edge : g.edges()){
            //            const auto uv = g.uv(edge); 
            //            out(edge,0) = uv.first;
            //            out(edge,1) = uv.second;
            //        }
            //        return out;
            //    }
            //)
            //.def("serialize",
            //    [](const GraphType & g) {
            //        nifty::marray::PyView<uint64_t> out({g.serializationSize()});
            //        auto ptr = &out(0);
            //        g.serialize(ptr);
            //        return out;
            //    }
            //)
            //.def("deserialize",
            //    [](GraphType & g, nifty::marray::PyView<uint64_t,1> serialization) {
            //        auto  startPtr = &serialization(0);
            //        auto  lastElement = &serialization(serialization.size()-1);
            //        auto d = lastElement - startPtr + 1;
            //        NIFTY_CHECK_OP(d,==,serialization.size(), "serialization must be contiguous");
            //        g.deserialize(startPtr);
            //    }
            //)
            //.def("extractSubgraphFromNodes",
            //    []( GraphType & g, const marray::PyView<int64_t,1> nodeList) {
            //        std::vector<int64_t> innerEdgesVec;  
            //        std::vector<int64_t> outerEdgesVec;  
            //        GraphType subgraph;
            //        {
            //            py::gil_scoped_release allowThreads;
            //            subgraph = g.extractSubgraphFromNodes(nodeList, innerEdgesVec, outerEdgesVec);
            //        }
            //        return std::make_tuple(innerEdgesVec, outerEdgesVec, subgraph);
            //    }
            //)
        ;

        // export the base graph API (others might derive)
        exportUndirectedGraphClassAPI<GraphType>(module, graphCls,clsName);


    }

    void exportUndirectedGridGraph(py::module & module){
        exportUndirectedGridGraphT<2>(module);
        exportUndirectedGridGraphT<3>(module);
    }
}
}
