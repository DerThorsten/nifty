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
#include "nifty/graph/rag/feature_accumulation/grid_rag_long_range_adjacency.hxx"

namespace py = pybind11;

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

    void exportLongRangeFeatures(py::module & ragModule) {

        typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d;
        exportGetLongRangeAdjacencyT<Rag3d>(ragModule);

    }

}
}
