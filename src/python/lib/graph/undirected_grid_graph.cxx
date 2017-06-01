#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


#include "boost/format.hpp"

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

        auto graphCls = py::class_<GraphType>(module, clsName.c_str(),
            (boost::format("%dDimensional Grid Graph")%DIM).str().c_str()
        );

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

            .def("imageToEdgeMap",
                [](
                    const GraphType & g,
                    nifty::marray::PyView<float, DIM> image,
                    const std::string & functorType
                ){

                    nifty::marray::PyView<float> out({g.edgeIdUpperBound()+1});

                    if(functorType == std::string("min")){
                        struct {
                            double operator()(const float a, const float b){
                                return std::min(a,b);
                            }
                        } op;
                        g.imageToEdgeMap(image, op, out);
                    }
                    else if(functorType == std::string("max")){
                        struct {
                            double operator()(const float a, const float b){
                                return std::max(a,b);
                            }
                        } op;
                        g.imageToEdgeMap(image, op, out);
                    }
                    else if(functorType == std::string("sum")){
                        struct {
                            double operator()(const float a, const float b){
                                return a + b;
                            }
                        } op;
                        g.imageToEdgeMap(image, op, out);
                    }
                    else if(functorType == std::string("prod")){
                        struct {
                            double operator()(const float a, const float b){
                                return a*b;
                            }
                        } op;
                        g.imageToEdgeMap(image, op, out);
                    }
                    else if(functorType == std::string("interpixel")){
                        g.imageToInterpixelEdgeMap(image, out);
                    }
                    else{
                        const auto s = boost::format("'%s' is an unknown mode. Must be in "
                            "['min', 'max', 'sum', 'prod', 'interpixel']")%functorType;
                        throw std::runtime_error(s.str());
                    }
                    return out;
                },
                py::arg("image"),
                py::arg("mode"),
                "convert an image to an edge map\n\n"
                "Arguments:\n\n"
                "   image (numpy.ndarray): the image\n"
                "    mode str: mode can be:\n"
                "       *   'min':  Minimum of the two image values at edges endpoints of coordinates.\n"
                "       *   'max':  Maximum of the two image values at edges endpoints of coordinates.\n"
                "       *   'sum':      Sum of the two image values at edges endpoints of coordinates.\n"
                "       *   'prod': Product of the two image values at edges endpoints of coordinates.\n"
            )


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
