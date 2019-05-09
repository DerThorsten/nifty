#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/project_to_pixels.hxx"
#include "nifty/graph/rag/project_to_pixels_stacked.hxx"

// still need this for python bindings of nifty::ArrayExtender
#include "nifty/python/converter.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class LABELS, class T, std::size_t DATA_DIM>
    void exportProjectScalarNodeDataToPixelsT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const GridRag<DATA_DIM, LABELS> & rag,
                const xt::pytensor<T, 1> & nodeData,
                const int numberOfThreads
           ){
                typedef typename xt::pytensor<T, DATA_DIM>::shape_type ShapeType;
                ShapeType shape;
                std::copy(rag.shape().begin(), rag.shape().end(), shape.begin());
                xt::pytensor<T, DATA_DIM> pixelData(shape);
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
                return pixelData;
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("numberOfThreads")=-1
        );
    }


    template<class LABELS, class T, class PIXEL_DATA, std::size_t DATA_DIM>
    void exportProjectScalarNodeDataToPixelsOutOfCoreT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const GridRag<DATA_DIM, LABELS> & rag,
                const xt::pytensor<T, 1> & nodeData,
                PIXEL_DATA & pixelData,
                nifty::array::StaticArray<int64_t, DATA_DIM> blockShape,
                const int numberOfThreads
           ){
                py::gil_scoped_release allowThreads;
                projectScalarNodeDataToPixelsOutOfCore(rag, nodeData, pixelData,
                                                       blockShape, numberOfThreads);
           },
           py::arg("graph"),
           py::arg("nodeData"),
           py::arg("pixelData"),
           py::arg("blockShape"),
           py::arg("numberOfThreads")=-1
        );
    }


    template<class LABELS, class T>
    void exportProjectScalarNodeDataToPixelsStackedT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const GridRagStacked2D<LABELS> & rag,
                const xt::pytensor<T, 1> & nodeData,
                const int numberOfThreads
           ){
                typedef typename xt::pytensor<T, 3>::shape_type ShapeType;
                ShapeType shape;
                std::copy(rag.shape().begin(), rag.shape().end(), shape.begin());
                xt::pytensor<T, 3> pixelData(shape);
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
                return pixelData;
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("numberOfThreads")=-1
        );
    }


    template<class LABELS, class T, class PIXEL_DATA>
    void exportProjectScalarNodeDataToPixelsStackedOutOfCoreT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const GridRagStacked2D<LABELS> & rag,
                const xt::pytensor<T, 1> & nodeData,
                PIXEL_DATA & pixelData,
                const int numberOfThreads
           ){
                const auto & shape = rag.shape();

                for(int d = 0; d < 3; ++d)
                    NIFTY_CHECK_OP(shape[d], ==, pixelData.shape()[d],
                                   "OutShape and Rag shape do not match!")
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
           },
           py::arg("graph"),
           py::arg("nodeData"),
           py::arg("pixelData"),
           py::arg("numberOfThreads")=-1
        );
    }


    template<class LABELS, class T, class PIXEL_DATA>
    void exportProjectScalarNodeDataInSubBlockT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataInSubBlock",
           [](
                const GridRagStacked2D<LABELS> & rag,
                const std::map<T, T> & nodeData,
                PIXEL_DATA & pixelData,
                const std::vector<int64_t> & blockBegin,
                const std::vector<int64_t> & blockEnd,
                const int numberOfThreads
           ){
                const auto & shape = rag.shape();
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataInSubBlock(rag, nodeData, pixelData,
                                                    blockBegin, blockEnd,
                                                    numberOfThreads);
                }
           },
           py::arg("rag"),
           py::arg("nodeData"),
           py::arg("pixelData"),
           py::arg("blockBegin"),
           py::arg("blockEnd"),
           py::arg("numberOfThreads")=-1
        );
    }


    void exportProjectToPixels(py::module & ragModule) {

        // exportScalarNodeDataToPixels
        {
            typedef xt::pytensor<uint32_t, 2> ExplicitPyLabels2D;
            typedef xt::pytensor<uint32_t, 3> ExplicitPyLabels3D;

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, uint32_t, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, uint32_t, 3>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, uint64_t, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, uint64_t, 3>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, float, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, float, 3>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, double, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, double, 3>(ragModule);
        }

        // z5
        #ifdef WITH_Z5
        {
            typedef nifty::nz5::DatasetWrapper<uint64_t> LabelsUInt64;
            exportProjectScalarNodeDataToPixelsOutOfCoreT<LabelsUInt64, uint64_t,
                                                          LabelsUInt64, 3>(ragModule);
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
