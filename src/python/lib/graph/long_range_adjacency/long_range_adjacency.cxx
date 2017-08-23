#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

// no hdf5 support for now
//#ifdef WITH_HDF5
//#include "nifty/hdf5/hdf5_array.hxx"
//#include "nifty/graph/rag/grid_rag_hdf5.hxx"
//#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
//#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
//#endif

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/feature_accumulation/grid_rag_long_range_features.hxx"

namespace py = pybind11;

// FIXME this doesn't really need the rag at all ....
namespace nifty{
namespace graph{

    using namespace py;

    template<class RAG>
    void exportGetLongRangeAdjacencyT(
        py::module & ragModule
    ){
        ragModule.def("getLongRangeAdjacency",
        [](
            const RAG & rag,
            const size_t longRange,
            const int numberOfThreads
        ){
            typedef typename RAG::LabelType LabelType;
            std::vector<std::pair<LabelType, LabelType>> adjacencyOut;
            {
                py::gil_scoped_release allowThreads;
                getLongRangeAdjacency(rag, longRange, adjacencyOut, numberOfThreads);
            }
            return adjacencyOut;
        },
        py::arg("rag"), py::arg("longRange"), py::arg("numberOfThreads")=-1
        );
    }

    template<class RAG>
    void exportAccumulateLongRangeFeaturesT(py::module & ragModule) {
        ragModule.def("accumulateLongRangeFeatures",
        [](
            const RAG & rag,
            const marray::PyView<float> affinities,
            const std::vector<std::pair<typename RAG::LabelType, typename RAG::LabelType>> & adjacency,
            const size_t longRange,
            const int numberOfThreads
        ){
            typedef typename RAG::LabelType LabelType;
            size_t nStats = 9;
            nifty::marray::PyView<float> features({adjacency.size(), nStats});
            {
                py::gil_scoped_release allowThreads;
                accumulateLongRangeFeatures(rag, affinities, adjacency, longRange, numberOfThreads);
            }
            return features;
        },
        py::arg("rag"), py::arg("affinities"), py::arg("adjacency"), py::arg("longRange"), py::arg("numberOfThreads")=-1
        );

    }

    void exportLongRangeFeatures(py::module & ragModule) {
        typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d;
        exportGetLongRangeAdjacencyT<Rag3d>(ragModule);
        exportAccumulateLongRangeFeaturesT<Rag3d>(ragModule);
    }

}
}
