#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// no hdf5 support for now
//#ifdef WITH_HDF5
//#include "nifty/hdf5/hdf5_array.hxx"
//#include "nifty/graph/rag/grid_rag_hdf5.hxx"
//#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
//#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
//#endif

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels_proxy.hxx"
#include "nifty/graph/rag/feature_accumulation/grid_rag_accumulate_flat.hxx"

#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

namespace nifty{
namespace graph{

    using namespace py;

    template<class RAG, class DATA_T>
    void exportAccumulateEdgeFeaturesFlatT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeFeaturesFlat",
        [](
            const RAG & rag,
            xt::pytensor<DATA_T, 3> data,
            const double minVal,
            const double maxVal,
            const int zDirection,
            const int numberOfThreads
        ){
            typedef xt::pytensor<DATA_T, 2> NumpyArrayType;
            NumpyArrayType edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeFeaturesFlat(rag, data, minVal, maxVal, edgeOut, zDirection, numberOfThreads);
            }
            return edgeOut;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("zDirection"),
        py::arg("numberOfThreads")= -1
        );
    }

    void exportAccumulateFlat(py::module & ragModule) {

        typedef LabelsProxy<3, xt::pytensor<uint32_t, 3>> ExplicitPyLabels3D;
        typedef GridRag<3, ExplicitPyLabels3D> Rag3d;
        exportAccumulateEdgeFeaturesFlatT<Rag3d, float>(ragModule);

    }

}
}
