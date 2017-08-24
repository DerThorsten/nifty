#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

#include "nifty/graph/long_range_adjacency/long_range_adjacency.hxx"
#include "nifty/graph/long_range_adjacency/accumulate_long_range_features.hxx"

namespace py = pybind11;

namespace nifty{
namespace graph{

    using namespace py;

    template<class ADJACENCY>
    void exportLongRangeFeaturesInCoreT(py::module & module) {
        ragModule.def("longRangeFeatures",
        [](
            const ADJACENCY & longRangeAdjacency,
            marray::PyView<typename ADJACENCY::LabelType, 3> labels,
            marray::PyView<float, 4> affinities,
            const int numberOfThreads
        ){
            size_t nStats = 9;
            nifty::marray::PyView<float> features({adjacency.size(), nStats});
            {
                py::gil_scoped_release allowThreads;
                longRangeFeatures(longRangeAdjacency, labels, affinities, numberOfThreads);
            }
            return features;
        },
        py::arg("longRangeAdjacency"), py::arg("labels"), py::arg("affinities"), py::arg("numberOfThreads")=-1
        );

    }

    // TODO HDF5 out of core

    void exportLongRangeFeatures(py::module & module) {
        typedef LongRangeAdjacency<marray::View<uint32_t>> ExplicitAdjacency;
        exportLongRangeFeaturesInCoreT<ExplicitAdjacency>(module);

        // TODO HDF5
    }

}
}
