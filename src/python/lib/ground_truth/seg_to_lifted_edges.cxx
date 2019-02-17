#include <iostream>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <xtensor-python/pytensor.hpp>

#include "nifty/ground_truth/seg_to_lifted_edges.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{


    void exportSeg2dToLiftedEdges(py::module & groundTruthModule){

        groundTruthModule.def("seg2dToLiftedEdges",
        [](
            const xt::pytensor<uint32_t, 2> & seg,
            const xt::pytensor<int32_t, 2> & edges
        ){
            NIFTY_CHECK_OP(edges.shape()[1], == , 2, "edges must be |N| x 2")
            xt::pytensor<uint8_t, 3> out = xt::zeros<uint8_t>({seg.shape()[0],
                                                               seg.shape()[1],
                                                               edges.shape()[0]});

            std::vector<std::array<int32_t, 2> > e(edges.shape()[0]);
            for(std::size_t i=0; i<e.size(); ++i){
                e[i][0] = edges(i, 0);
                e[i][1] = edges(i, 1);
            }

            seg2dToLiftedEdges(seg, e, out);

            return out;
        },
            py::arg("segmentation"),
            py::arg("edges")
        );
    }

    void exportSeg3dToLiftedEdges(py::module & groundTruthModule){

        groundTruthModule.def("seg2dToLiftedEdges",
        [](
            const xt::pytensor<uint32_t, 2> & seg,
            const xt::pytensor<int32_t, 2> & edges,
            const int32_t z
        ){
            NIFTY_CHECK_OP(edges.shape()[1], == , 3, "edges must be |N| x 3");
            xt::pytensor<uint8_t, 3> out = xt::zeros<uint8_t>({seg.shape()[0],
                                                               seg.shape()[1],
                                                               edges.shape()[0]});

            std::vector<std::array<int32_t, 3> > e(edges.shape()[0]);
            for(std::size_t i=0; i<e.size(); ++i){
                e[i][0] = edges(i, 0);
                e[i][1] = edges(i, 1);
                e[i][2] = edges(i, 2);
            }

            seg3dToLiftedEdges(seg, e, z, out);

            return out;
        },
            py::arg("segmentation"),
            py::arg("edges"),
            py::arg("z")
        );
    }


    void exportSeg3dToCremiZ5Edges(py::module & groundTruthModule){

        groundTruthModule.def("seg3dToCremiZ5Edges",
        [](
            const xt::pytensor<uint32_t, 3> &  seg,
            const xt::pytensor<int32_t, 2> & edges
        ){
            NIFTY_CHECK_OP(edges.shape()[1], == , 4, "edges must be |N| x 4");

            xt::pytensor<uint8_t, 3> out = xt::zeros<uint8_t>({seg.shape()[0],
                                                               seg.shape()[1],
                                                               edges.shape()[0]});

            std::vector<std::array<int32_t, 4> > e(edges.shape()[0]);
            for(std::size_t i=0; i<e.size(); ++i){
                e[i][0] = edges(i, 0);
                e[i][1] = edges(i, 1);
                e[i][2] = edges(i, 2);
                e[i][2] = edges(i, 3);
            }

            seg3dToCremiZ5Edges(seg, e, out);

            return out;
        },
            py::arg("segmentation"),
            py::arg("edges")
        );

        #if 0
        groundTruthModule.def("seg3dToCremiZ5Edges",
        [](
            xt::pytensor<uint32_t, 3>   seg,
            xt::pytensor<uint32_t, 3>   dt2d,
            xt::pytensor<int32_t, 2 >   edges,
            xt::pytensor<float, 1 >    edge_priors,
        ){
            NIFTY_CHECK_OP(edges.shape(1), == , 4, "edges must be |N| x 4");

            xt::pytensor<uint8_t> out({
                std::size_t(seg.shape(0)),
                std::size_t(seg.shape(1)),
                std::size_t(edges.shape(0))
            });

            xt::pytensor<float> w({
                std::size_t(seg.shape(0)),
                std::size_t(seg.shape(1)),
                std::size_t(edges.shape(0))
            });
            std::vector<std::array<int32_t, 4> > e(edges.shape(0));
            std::vector<float>                   p(edges.shape(0));
            for(std::size_t i=0; i<e.size(); ++i){
                p[i] = edge_priors()
                e[i][0] = edges(i, 0);
                e[i][1] = edges(i, 1);
                e[i][2] = edges(i, 2);
                e[i][2] = edges(i, 3);
            }

            seg3dToCremiZ5Edges(seg, dt2d, e, out, w);

            std::make_pair(out,w);
        },
            py::arg("segmentation"),
            py::arg("distance_transform_2d"),
            py::arg("edges"),
            py::arg("edge_priors")
        );
        #endif
    }

    void exportThinSegFilter(py::module & groundTruthModule){

        groundTruthModule.def("_thinSegFilter",
        [](
            const xt::pytensor<uint32_t, 2> & seg,
            xt::pytensor<uint32_t, 2 > & dt,
            const float sigma,
            const int r
        ){
            xt::pytensor<float, 2> out = xt::zeros<float>(seg.shape());
            thinSegFilter(seg, dt, out, sigma, r);
            return out;
        },
            py::arg("seg"),
            py::arg("dt"),
            py::arg("sigma"),
            py::arg("radius") = 0
        );


        #if 0
        groundTruthModule.def("seg3dToCremiZ5Edges",
        [](
            xt::pytensor<uint32_t, 3>   seg,
            xt::pytensor<uint32_t, 3>   dt2d,
            xt::pytensor<int32_t, 2>   edges,
            xt::pytensor<float, 1>    edge_priors,
        ){
            NIFTY_CHECK_OP(edges.shape(1), == , 4, "edges must be |N| x 4");

            xt::pytensor<uint8_t> out({
                std::size_t(seg.shape(0)),
                std::size_t(seg.shape(1)),
                std::size_t(edges.shape(0))
            });

            xt::pytensor<float> w({
                std::size_t(seg.shape(0)),
                std::size_t(seg.shape(1)),
                std::size_t(edges.shape(0))
            });

            std::vector<std::array<int32_t, 4> > e(edges.shape(0));
            std::vector<float>                   p(edges.shape(0));
            for(std::size_t i=0; i<e.size(); ++i){
                p[i] = edge_priors()
                e[i][0] = edges(i, 0);
                e[i][1] = edges(i, 1);
                e[i][2] = edges(i, 2);
                e[i][2] = edges(i, 3);
            }

            seg3dToCremiZ5Edges(seg, dt2d, e, out, w);

            std::make_pair(out,w);
        },
            py::arg("segmentation"),
            py::arg("distance_transform_2d"),
            py::arg("edges"),
            py::arg("edge_priors")
        );
        #endif
    }


    void exportSegToLiftedEdges(py::module & groundTruthModule){
        exportSeg2dToLiftedEdges(groundTruthModule);
        exportSeg3dToLiftedEdges(groundTruthModule);
        exportSeg3dToCremiZ5Edges(groundTruthModule);
        exportThinSegFilter(groundTruthModule);
    }
}
}
