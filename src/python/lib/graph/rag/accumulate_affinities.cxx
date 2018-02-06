#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/feature_accumulation/grid_rag_affinity_features.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;

    template<class RAG, unsigned DIM>
    void exportAccumulateAffinityFeaturesT(
        py::module & ragModule
    ){
        // features for lifted nh not supported right now
        /*
        ragModule.def("computeFeaturesAndNhFromAffinities",
        [](
            const RAG & rag,
            xt::pytensor<float, DIM+1> affinities,
            const std::vector<std::vector<int>> & offsets,
            const int numberOfThreads
        ){

            // TODO
            //{
            //    py::gil_scoped_release allowThreads;
            //}
            LiftedNh<RAG> lnh(
                rag, offsets.begin(), offsets.end(), numberOfThreads
            );

            int64_t nLocal  = rag.edgeIdUpperBound() + 1;
            int64_t nLifted = lnh.edgeIdUpperBound() + 1;
            xt::pytensor<float, 2> outLocal({nLocal, int64_t(10)});
            xt::pytensor<float, 2> outLifted({nLifted, int64_t(10)});
            {
                py::gil_scoped_release allowThreads;
                accumulateLongRangeAffinities(rag, lnh, affinities, 0., 1.,  outLocal, outLifted, numberOfThreads);
            }
            xt::pytensor<uint32_t, 2> lnhOut({nLifted, int64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                for(size_t e = 0; e < nLifted; ++e) {
                    lnhOut(e, 0) = lnh.u(e);
                    lnhOut(e, 1) = lnh.v(e);
                }
            }
            return std::make_tuple(lnhOut, outLocal, outLifted);
        },
        py::arg("rag"),
        py::arg("affinities"),
        py::arg("offsets"),
        py::arg("numberOfThreads")= -1
        );
        */

        ragModule.def("featuresFromLocalAffinities",
        [](
            const RAG & rag,
            const xt::pytensor<float, DIM+1> & affinities,
            const std::vector<std::array<int, 3>> & offsets,
            const float min, const float max,
            const int numberOfThreads
        ){

            int64_t nEdges = rag.numberOfEdges();
            typename xt::pytensor<float, 2>::shape_type shape = {nEdges, int64_t(9)};
            xt::pytensor<float, 2> out(shape);
            {
                py::gil_scoped_release allowThreads;
                accumulateAffinities(rag, affinities, offsets, out, min, max, numberOfThreads);
            }
            return out;
        }, py::arg("rag"),
           py::arg("affinities"),
           py::arg("offsets"),
           py::arg("min")=0., py::arg("max")=0.,
           py::arg("numberOfThreads")=-1
        );
    }

    void exportAccumulateAffinityFeatures(py::module & ragModule) {
        typedef xt::pytensor<uint32_t, 3> ExplicitPyLabels3D;
        typedef GridRag<3, ExplicitPyLabels3D> Rag3d;
        exportAccumulateAffinityFeaturesT<Rag3d, 3>(ragModule);
    }

}
}
