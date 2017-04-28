#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

// now hdf5 support for now
//#ifdef WITH_HDF5
//#include "nifty/hdf5/hdf5_array.hxx"
//#include "nifty/graph/rag/grid_rag_hdf5.hxx"
//#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
//#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
//#endif

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_coordinates.hxx"

namespace py = pybind11;

namespace nifty{
namespace graph{
    
    using namespace py;

    //struct PyCoordinateVectorType : public CoordinateVectorType {
    //};

    //void exportCoordinateVectorType(py::module & module) {
    //    module.class_<>
    //}

    // TODO export CoordinateVectorType properly to not rely on
    // pybind::stl magic ?!
    template<class RAG>
    void exportGridRagCoordinatesT(py::module & module) {
        module.def("edgeCoordinatesImpl",
        [](
           const RAG & rag,
           const int numberOfThreads
        ){
            CoordinateVectorType out;
            {
                py::gil_scoped_release allowThreads;
                computeEdgeCoordinates(rag, out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"), py::arg("numberOfThreads") = -1
        ); 

    }

    void exportGridRagCoordinates(py::module & module) {
        typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d;
        exportGridRagCoordinatesT<Rag3d>(module);
    }

}
}
