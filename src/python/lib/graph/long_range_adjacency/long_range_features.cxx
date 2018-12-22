#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

#include "nifty/graph/long_range_adjacency/long_range_adjacency.hxx"
#include "nifty/graph/long_range_adjacency/accumulate_long_range_features.hxx"
#include "nifty/tools/runtime_check.hxx"

namespace py = pybind11;

namespace nifty{
namespace graph{

    using namespace py;

    template<class ADJACENCY>
    void exportLongRangeFeaturesInCoreT(py::module & module) {
        module.def("longRangeFeatures",
        [](
            const ADJACENCY & longRangeAdjacency,
            const xt::pytensor<typename ADJACENCY::LabelType, 3> & labels,
            const xt::pytensor<float, 4> & affinities,
            const int zDirection,
            const int numberOfThreads
        ){
            NIFTY_CHECK_OP(affinities.shape()[0], ==,
                           longRangeAdjacency.range()-1,
                           "Number of channels is wrong!");
            for(int d = 0; d < 3; ++d) {
                NIFTY_CHECK_OP(affinities.shape()[d + 1], ==,
                               longRangeAdjacency.shape()[d], "Wrong shape");
            }
            size_t nStats = 9;
            xt::pytensor<float, 2> features({int64_t(longRangeAdjacency.numberOfEdges()),
                                             int64_t(nStats)});
            {
                py::gil_scoped_release allowThreads;
                accumulateLongRangeFeatures(longRangeAdjacency, labels,
                                            affinities, features,
                                            zDirection, numberOfThreads);
            }
            return features;
        },
        py::arg("longRangeAdjacency"),
        py::arg("labels"),
        py::arg("affinities"),
        py::arg("zDirection"),
        py::arg("numberOfThreads")=-1
        );

    }

    // TODO HDF5 out of core
    void exportLongRangeFeatures(py::module & module) {
        typedef xt::pytensor<uint32_t, 3> ExplicitLabels;
        typedef LongRangeAdjacency<ExplicitLabels> ExplicitAdjacency;
        exportLongRangeFeaturesInCoreT<ExplicitAdjacency>(module);
    }

}
}
