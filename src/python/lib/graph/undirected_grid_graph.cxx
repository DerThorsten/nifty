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

            .def("affinitiesToEdgeMap",
                [](
                    const GraphType & g,
                    xt::pytensor<float, DIM + 1> affinities
                ){
                    typedef typename xt::pytensor<float, 1>::shape_type ShapeType;
                    ShapeType shape = {g.edgeIdUpperBound() + 1};
                    xt::pytensor<float, 1> out = xt::zeros<float>(shape);
                    g.affinitiesToEdgeMap(affinities, out);
                    return out;
                },
                py::arg("affinities")
            )

            .def("liftedProblemFromLongRangeAffinities",
                [](const GraphType & g,
                   xt::pytensor<float, DIM+1> affinities,
                   const std::vector<std::vector<int>> & offsets) {

                    // upper bound for the number of lifted edges
                    // we assume that first DIM channels are direct nhood channels
                    const auto & shape = affinities.shape();
                    int64_t nLiftedTot = (shape[0] - DIM) * std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<int64_t>());

                    // initialize all the output
                    typedef typename xt::pytensor<float, 1>::shape_type ShapeType;
                    ShapeType localShape = {static_cast<int64_t>(g.edgeIdUpperBound() + 1)};
                    ShapeType liftedShape = {nLiftedTot};
                    typedef typename xt::pytensor<uint64_t, 2>::shape_type UvShape;
                    UvShape uvShape = {nLiftedTot, 2};

                    xt::pytensor<float, 1> localFeatures(localShape);
                    xt::pytensor<float, 1> liftedFeatures(liftedShape);
                    xt::pytensor<uint64_t, 2> liftedUvs(uvShape);
                    int64_t nLifted;
                    {
                        py::gil_scoped_release allowThreads;
                        nLifted = g.longRangeAffinitiesToLiftedEdges(affinities,
                                                                     localFeatures,
                                                                     liftedUvs,
                                                                     liftedFeatures,
                                                                     offsets);
                    }
                    // FIXME resizing zeros out everything
                    // ShapeType actualLiftedShape = {nLifted};
                    // liftedFeatures.resize(actualLiftedShape);
                    // UvShape actualUvShape = {nLifted, 2};
                    // liftedUvs.resize(actualUvShape);
                    return std::make_tuple(nLifted, localFeatures, liftedUvs, liftedFeatures);
                },
                py::arg("affinities"), py::arg("offsets")
            )

            .def("liftedProblemFromLongRangeAffinitiesWithStrides",
                [](const GraphType & g,
                   xt::pytensor<float, DIM+1> affinities,
                   const std::vector<std::vector<int>> & offsets,
                   const std::vector<int> & strides) {

                    // upper bound for the number of lifted edges
                    // we assume that first DIM channels are direct nhood channels
                    const auto & shape = affinities.shape();
                    int64_t nLiftedTot = (shape[0] - DIM) * std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<int64_t>());

                    // initialize all the output
                    typedef typename xt::pytensor<float, 1>::shape_type ShapeType;
                    ShapeType localShape = {static_cast<int64_t>(g.edgeIdUpperBound() + 1)};
                    ShapeType liftedShape = {nLiftedTot};
                    typedef typename xt::pytensor<uint64_t, 2>::shape_type UvShape;
                    UvShape uvShape = {nLiftedTot, 2};

                    xt::pytensor<float, 1> localFeatures = xt::zeros<float>(localShape);
                    xt::pytensor<float, 1> liftedFeatures = xt::zeros<float>(liftedShape);
                    xt::pytensor<uint64_t, 2> liftedUvs(uvShape);
                    int64_t nLifted;
                    {
                        py::gil_scoped_release allowThreads;
                        nLifted = g.longRangeAffinitiesToLiftedEdges(affinities,
                                                                     localFeatures,
                                                                     liftedUvs,
                                                                     liftedFeatures,
                                                                     offsets,
                                                                     strides);
                    }

                    // FIXME resize zeros out everything
                    // ShapeType actualLiftedShape = {nLifted};
                    // liftedFeatures.resize(actualLiftedShape);
                    // UvShape actualUvShape = {nLifted, 2};
                    // liftedUvs.resize(actualUvShape);

                    return std::make_tuple(nLifted, localFeatures, liftedUvs, liftedFeatures);
                },
                py::arg("affinities"), py::arg("offsets"), py::arg("strides")
            )
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
