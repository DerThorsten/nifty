#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/project_to_pixels.hxx"
#include "nifty/graph/rag/project_to_pixels_stacked.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class LABELS_PROXY, class T, std::size_t DATA_DIM>
    void exportProjectScalarNodeDataToPixelsT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const GridRag<DATA_DIM, LABELS_PROXY> & rag,
                const xt::pytensor<T, 1> nodeData,
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


    template<class LABELS_PROXY, class T>
    void exportProjectScalarNodeDataToPixelsStackedExplicitT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const GridRagStacked2D<LABELS_PROXY> & rag,
                const xt::pytensor<T, 1> nodeData,
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


    #ifdef WITH_HDF5
    template<class LABELS_PROXY, class T>
    void exportProjectScalarNodeDataToPixelsStackedHdf5T(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const GridRagStacked2D<LABELS_PROXY> & rag,
                const xt::pytensor<T, 1> nodeData,
                nifty::hdf5::Hdf5Array<T> & pixelData,
                const int numberOfThreads
           ){
                const auto & shape = rag.shape();

                for(int d = 0; d < 3; ++d)
                    NIFTY_CHECK_OP(shape[d],==,pixelData.shape(d),"OutShape and Rag shape do not match!")

                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("pixelData"),py::arg("numberOfThreads")=-1
        );
    }


    template<class LABELS_PROXY, class T>
    void exportProjectScalarNodeDataInSubBlockStackedHdf5T(py::module & ragModule){

        ragModule.def("projectScalarNodeDataInSubBlock",
           [](
                const GridRagStacked2D<LABELS_PROXY> & rag,
                const std::map<T, T> & nodeData,
                nifty::hdf5::Hdf5Array<T>  & pixelData,
                const std::vector<int64_t> & blockBegin,
                const std::vector<int64_t> & blockEnd,
                const int numberOfThreads
           ){
                const auto & shape = rag.shape();
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataInSubBlock(
                        rag, nodeData, pixelData, blockBegin, blockEnd, numberOfThreads
                    );
                }
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("pixelData"),py::arg("blockBegin"),py::arg("blockEnd"),py::arg("numberOfThreads")=-1
        );
    }
    #endif


    void exportProjectToPixels(py::module & ragModule) {

        // exportScalarNodeDataToPixels
        {
            typedef LabelsProxy<2, xt::pytensor<uint32_t, 2>> ExplicitPyLabels2D;
            typedef LabelsProxy<3, xt::pytensor<uint32_t, 3>> ExplicitPyLabels3D;

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, uint32_t, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, uint32_t, 3>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, uint64_t, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, uint64_t, 3>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, float, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, float, 3>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels2D, double, 2>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitPyLabels3D, double, 3>(ragModule);
        }

        // exportScalarNodeDataToPixelsStacked
        {
            // explicit
            {
                typedef LabelsProxy<3, xt::pytensor<uint32_t, 3>> LabelsUInt32;

                exportProjectScalarNodeDataToPixelsStackedExplicitT<LabelsUInt32, uint32_t>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<LabelsUInt32, uint64_t>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<LabelsUInt32, float>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<LabelsUInt32, double>(ragModule);
            }

            // hdf5
            #ifdef WITH_HDF5
            {
                typedef Hdf5Labels<3,uint32_t> LabelsUInt32;
                typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;

                exportProjectScalarNodeDataToPixelsStackedHdf5T<LabelsUInt32, uint32_t>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<LabelsUInt32, uint64_t>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<LabelsUInt32, float>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<LabelsUInt32, double>(ragModule);

                exportProjectScalarNodeDataInSubBlockStackedHdf5T<LabelsRagUInt32, uint32_t>(ragModule);
                exportProjectScalarNodeDataInSubBlockStackedHdf5T<LabelsRagUInt32, uint64_t>(ragModule);
            }
            #endif
        }
    }

} // end namespace graph
} // end namespace nifty
