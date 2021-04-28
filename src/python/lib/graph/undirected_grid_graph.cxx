#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"
#include "boost/format.hpp"

#include "nifty/graph/undirected_grid_graph.hxx"
#include "export_undirected_graph_class_api.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    template<std::size_t DIM>
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
            .def_property_readonly("shape", [](const GraphType & g){
                std::vector<uint64_t> shape_(DIM);
                for(int d = 0; d < DIM; ++d) {
                    shape_[d] = g.shape(d);
                }
                return shape_;
            })
            .def("nodeToCoordinate",[](
                const GraphType & g,
                const uint64_t node
            ){
                return g.nodeToCoordinate(node);
            })
            .def("coordinateToNode",[](
                const GraphType & g,
                const typename GraphType::CoordinateType & coord
            ){
                return g.coordinateToNode(coord);
            })

            .def("euclideanEdgeMap",
                [](
                    const GraphType & g,
                    const xt::pytensor<bool, DIM> & mask,
                    const std::array<double, DIM> & resolution
                ){
                    typedef typename GraphType::CoordinateType CoordinateType;
                    xt::pytensor<float, 1> out = xt::zeros<float>({g.edgeIdUpperBound() + 1});
                    for(const int64_t edge : g.edges()){

                        const auto & uv = g.uv(edge);
                        CoordinateType cU, cV;
                        g.nodeToCoordinate(uv.first,  cU);
                        g.nodeToCoordinate(uv.second, cV);
                        const bool uVal = xtensor::read(mask, cU.asStdArray());
                        const bool vVal = xtensor::read(mask, cU.asStdArray());

                        // if one or more of the values is outside of the mask,
                        // the edge is not allowed
                        if(uVal == 0 || vVal == 0) {
                            out[edge] = std::numeric_limits<float>::infinity();
                            continue;
                        }

                        // euclidean distance
                        double dist = 0;
                        double aux;
                        for(unsigned d = 0; d < DIM; ++d) {
                            aux = (cU[d] - cV[d]) * resolution[d];
                            dist += aux * aux;
                        }
                        out[edge] = sqrt(dist);
                    }
                    return out;
                }
            )

            .def("imageToEdgeMap",
                [](
                    const GraphType & g,
                    const xt::pytensor<float, DIM> & image,
                    const std::string & functorType
                ){

                    typedef typename xt::pytensor<float, 1>::shape_type ShapeType;
                    ShapeType shape = {g.edgeIdUpperBound() + 1};
                    xt::pytensor<float, 1> out(shape);

                    {
                        py::gil_scoped_release allowThreads;
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
                        else if(functorType == std::string("l1")){
                            struct {
                                double operator()(const float a, const float b){
                                    return std::abs(a - b);
                                }
                            } op;
                            g.imageToEdgeMap(image, op, out);
                        }
                        else if(functorType == std::string("l2")){
                            struct {
                                double operator()(const float a, const float b){
                                    return (a - b) * (a - b);
                                }
                            } op;
                            g.imageToEdgeMap(image, op, out);
                        }
                        else if(functorType == std::string("interpixel")){
                            g.imageToInterpixelEdgeMap(image, out);
                        }
                        else{
                            const auto s = boost::format("'%s' is an unknown mode. Must be in "
                                "['min', 'max', 'sum', 'prod', 'interpixel', 'l1', 'l2']")%functorType;
                            throw std::runtime_error(s.str());
                        }
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

            .def("imageWithChannelsToEdgeMap",
                [](
                    const GraphType & g,
                    const xt::pytensor<float, DIM + 1> & image,
                    const std::string & distance
                ){

                    typedef typename xt::pytensor<float, 1>::shape_type ShapeType;
                    ShapeType shape = {g.edgeIdUpperBound() + 1};
                    xt::pytensor<float, 1> out(shape);
                    {
                        py::gil_scoped_release allowThreads;
                        g.imageWithChannelsToEdgeMap(image, distance, out);
                    }
                    return out;
                },
                py::arg("image"),
                py::arg("distance")
            )

            .def("imageWithChannelsToEdgeMapWithOffsets",
                [](
                    const GraphType & g,
                    const xt::pytensor<float, DIM + 1> & image,
                    const std::string & distance,
                    const std::vector<std::vector<int>> & offsets,
                    const std::optional<std::vector<int>> & strides,
                    const bool randomize_strides
                ){
                    // upper bound for the number of edges
                    const auto & shape = image.shape();
                    int64_t nEdges = offsets.size() * std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<int64_t>());

                    typedef typename xt::pytensor<uint64_t, 2>::shape_type EdgeShape;
                    EdgeShape edgeShape = {nEdges, 2};
                    xt::pytensor<uint64_t, 2> edges(edgeShape);

                    typedef typename xt::pytensor<float, 1>::shape_type EdgeMapShape;
                    EdgeMapShape edgeMapShape = {nEdges};
                    xt::pytensor<float, 1> edgeMap(edgeMapShape);

                    {
                        py::gil_scoped_release allowThreads;
                        if(strides.has_value() && randomize_strides) {
                            auto & strides_val = strides.value();
                            const double p_sample = 1. / std::accumulate(strides_val.begin(), strides_val.end(), 1., std::multiplies<double>());
                            nEdges = g.imageWithChannelsToEdgeMapWithOffsets(image, distance, offsets, p_sample, edges, edgeMap);
                        } else if(strides.has_value()) {
                            nEdges = g.imageWithChannelsToEdgeMapWithOffsets(image, distance, offsets, strides.value(), edges, edgeMap);
                        } else {
                            nEdges = g.imageWithChannelsToEdgeMapWithOffsets(image, distance, offsets, edges, edgeMap);
                        }
                    }
                    return std::make_tuple(nEdges, edges, edgeMap);
                },
                py::arg("image"),
                py::arg("distance"),
                py::arg("offsets"),
                py::arg("strides")=std::nullopt,
                py::arg("randomize_strides")=false
            )

            .def("affinitiesToEdgeMap",
                [](
                    const GraphType & g,
                    const xt::pytensor<float, DIM + 1> & affinities,
                    const bool toLower
                ){
                    typedef typename xt::pytensor<float, 1>::shape_type ShapeType;
                    ShapeType shape = {g.edgeIdUpperBound() + 1};
                    xt::pytensor<float, 1> out = xt::zeros<float>(shape);
                    {
                        py::gil_scoped_release allowThreads;
                        g.affinitiesToEdgeMap(affinities, out, toLower);
                    }
                    return out;
                },
                py::arg("affinities"),
                py::arg("toLower")=true
            )

            .def("affinitiesToEdgeMapWithOffsets",
                [](
                    const GraphType & g,
                    const xt::pytensor<float, DIM + 1> & affinities,
                    const std::vector<std::vector<int>> & offsets,
                    const std::optional<std::vector<int>> & strides,
                    const bool randomize_strides
                ){
                    // upper bound for the number of edges
                    const auto & shape = affinities.shape();
                    int64_t nEdges = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());

                    typedef typename xt::pytensor<uint64_t, 2>::shape_type EdgeShape;
                    EdgeShape edgeShape = {nEdges, 2};
                    xt::pytensor<uint64_t, 2> edges(edgeShape);

                    typedef typename xt::pytensor<float, 1>::shape_type EdgeMapShape;
                    EdgeMapShape edgeMapShape = {nEdges};
                    xt::pytensor<float, 1> edgeMap(edgeMapShape);

                    {
                        py::gil_scoped_release allowThreads;
                        if(strides.has_value() && randomize_strides) {
                            auto & strides_val = strides.value();
                            const double p_sample = 1. / std::accumulate(strides_val.begin(), strides_val.end(), 1., std::multiplies<double>());
                            nEdges = g.affinitiesToEdgeMapWithOffsets(affinities,
                                                                      offsets,
                                                                      p_sample,
                                                                      edges,
                                                                      edgeMap);
                        } else if(strides.has_value()) {
                            nEdges = g.affinitiesToEdgeMapWithOffsets(affinities,
                                                                      offsets,
                                                                      strides.value(),
                                                                      edges,
                                                                      edgeMap);

                        } else {
                            nEdges = g.affinitiesToEdgeMapWithOffsets(affinities,
                                                                      offsets,
                                                                      edges,
                                                                      edgeMap);
                        }
                    }

                    // NOTE xtensor::resize does not work as expected and changes the values here
                    // return the number of actual edges instead and resize in python
                    /*
                    edgeShape = {nEdges, 2};
                    edges.resize(edgeShape);

                    edgeMapShape = {nEdges};
                    edgeMap.resize(edgeMapShape);
                    */

                    return std::make_tuple(nEdges, edges, edgeMap);

                },
                py::arg("affinities"),
                py::arg("offsets"),
                py::arg("strides")=std::nullopt,
                py::arg("randomize_strides")=false
            )

            .def("affinitiesToEdgeMapWithMask",
                [](
                    const GraphType & g,
                    const xt::pytensor<float, DIM + 1> & affinities,
                    const std::vector<std::vector<int>> & offsets,
                    const xt::pytensor<float, DIM + 1> & mask
                ){
                    // upper bound for the number of edges
                    const auto & shape = affinities.shape();
                    int64_t nEdges = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());

                    typedef typename xt::pytensor<uint64_t, 2>::shape_type EdgeShape;
                    EdgeShape edgeShape = {nEdges, 2};
                    xt::pytensor<uint64_t, 2> edges(edgeShape);

                    typedef typename xt::pytensor<float, 1>::shape_type EdgeMapShape;
                    EdgeMapShape edgeMapShape = {nEdges};
                    xt::pytensor<float, 1> edgeMap(edgeMapShape);

                    {
                        py::gil_scoped_release allowThreads;
                        nEdges = g.affinitiesToEdgeMapWithOffsets(affinities,
                                                                  offsets,
                                                                  mask,
                                                                  edges,
                                                                  edgeMap);
                    }

                    return std::make_tuple(nEdges, edges, edgeMap);

                },
                py::arg("affinities"),
                py::arg("offsets"),
                py::arg("mask")
            )

        .def("projectEdgeIdsToPixels",
            [](
                const GraphType & g
            ){
                typename xt::pytensor<int64_t, DIM+1>::shape_type retshape;
                retshape[0] = DIM;
                for(auto d=0; d<DIM; ++d){
                    retshape[d+1] = g.shape(d);
                }
                xt::pytensor<int64_t, DIM+1> ret(retshape);

                {
                    py::gil_scoped_release allowThreads;
                    g.projectEdgeIdsToPixels(ret);
                }
                return ret;
            }
        )

        .def("projectEdgeIdsToPixelsWithOffsets",
            [](
                const GraphType & g,
                const std::vector<std::vector<int>> & offsets,
                const std::optional<std::vector<int>> & strides,
                const std::optional<xt::pytensor<bool, DIM+1>> & mask

            ){
                typename xt::pytensor<int64_t, DIM+1>::shape_type retshape;
                retshape[0] = offsets.size();
                for(auto d=0; d<DIM; ++d){
                    retshape[d+1] = g.shape(d);
                }
                xt::pytensor<int64_t, DIM+1> ret(retshape);

                {
                    py::gil_scoped_release allowThreads;
                    if(strides.has_value() && mask.has_value()) {
                        throw std::runtime_error("Strides and mask together are not suported");
                    } else if(strides.has_value()) {
                        g.projectEdgeIdsToPixels(offsets, strides.value(), ret);
                    } else if(mask.has_value()) {
                        g.projectEdgeIdsToPixels(offsets, mask.value(), ret);
                    } else {
                        g.projectEdgeIdsToPixels(offsets, ret);
                    }
                }
                return ret;
            },
            py::arg("offsets"),
            py::arg("strides")=std::nullopt,
            py::arg("mask")=std::nullopt
        )

        .def("projectNodeIdsToPixels", [](const GraphType & g){
            typename xt::pytensor<uint64_t, DIM>::shape_type retshape;
            for(auto d=0; d<DIM; ++d){
                retshape[d] = g.shape(d);
            }
            xt::pytensor<uint64_t, DIM> ret(retshape);
            {
                py::gil_scoped_release allowThreads;
                g.projectNodeIdsToPixels(ret);
            }
            return ret;
        })
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
