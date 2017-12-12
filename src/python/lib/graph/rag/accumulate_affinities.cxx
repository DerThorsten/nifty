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

    // TODO
    // - parallelize properly
    // - lift gil everywhere
    // - return count of affinities per edge in features
    template<class RAG, unsigned DIM>
    void exportAccumulateAffinityFeaturesT(
        py::module & ragModule
    ){
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

        ragModule.def("featuresFromLocalAffinities",
        [](
            const RAG & rag,
            xt::pytensor<float, DIM+1> affinities,
            const int numberOfThreads
        ){

            int64_t nEdges  = rag.edgeIdUpperBound() + 1;
            xt::pytensor<float, 2> out({nEdges, int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateAffinities(rag, affinities, 0., 1.,  out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("affinities"),
        py::arg("numberOfThreads")= -1
        );
    }

    void exportAccumulateAffinityFeatures(py::module & ragModule) {
        typedef xt::pytensor<uint32_t, 3> ExplicitPyLabels3D;
        typedef GridRag<3, ExplicitPyLabels3D> Rag3d;
        // FIXME FIXME we need `unravelIndex` functionality for this
        //exportAccumulateAffinityFeaturesT<Rag3d, 3>(ragModule);
    }

}
}
