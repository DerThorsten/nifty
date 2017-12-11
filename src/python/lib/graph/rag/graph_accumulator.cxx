#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/graph/rag/grid_rag_features.hxx"
#include "nifty/graph/rag/grid_rag_features_stacked.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

#include "xtensor-python/pytensor.hpp"


namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;


    template<class RAG, class T, std::size_t DATA_DIM>
    void exportGridRagAccumulateLabelsT(py::module & ragModule){

        ragModule.def("gridRagAccumulateLabels",
            [](
                const RAG & rag,
                xt::pytensor<T, DATA_DIM> labels,
                const bool ignoreBackground,
                const T ignoreValue
            ){
                xt::pytensor<T, 1> nodeLabels = xt::zeros<T>({(int64_t) rag.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateLabels(rag, labels, nodeLabels, ignoreBackground, ignoreValue);
                }
                return nodeLabels;

            },
            py::arg("graph"),
            py::arg("labels"),
            py::arg("ignoreBackground")=false,
            py::arg("ignoreValue")=0
        );
    }

    template<class RAG, class DATA>
    void exportGridRagStackedAccumulateLabelsT(py::module & ragModule){

        ragModule.def("gridRagAccumulateLabels",
            [](
                const RAG & rag,
                DATA labels,
                const int numberOfThreads
            ){
                typedef typename DATA::value_type DataType;
                xt::pytensor<DataType, 1> nodeLabels = xt::zeros<DataType>({(int64_t) rag.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateLabels(rag, labels, nodeLabels);
                }
                return nodeLabels;

            },
            py::arg("graph"),
            py::arg("labels"),
            py::arg("numberOfThreads") = -1
        );
    }

    template<class RAG, class NODE_TYPE>
    void exportGetSkipEdgesForSliceT(
        py::module & ragModule
    ){
        ragModule.def("getSkipEdgesForSlice",
        [](
            const RAG & rag,
            const uint64_t z,
            std::map<size_t,std::vector<NODE_TYPE>> & defectNodes, // all defect nodes
            const bool lowerIsCompletelyDefected
        ){
            std::vector<size_t> deleteEdges;
            std::vector<size_t> ignoreEdges;

            std::vector<std::pair<NODE_TYPE,NODE_TYPE>> skipEdges;
            std::vector<size_t> skipRanges;
            {
                py::gil_scoped_release allowThreads;
                getSkipEdgesForSlice(
                    rag,
                    z,
                    defectNodes,
                    deleteEdges,
                    ignoreEdges,
                    skipEdges,
                    skipRanges,
                    lowerIsCompletelyDefected
                );
            }
            return std::make_tuple(deleteEdges, ignoreEdges, skipEdges, skipRanges);
        },
        py::arg("rag"),
        py::arg("z"),
        py::arg("defectNodes"),
        py::arg("lowerIsCompletelyDefected")
        );
    }

    void exportGraphAccumulator(py::module & ragModule) {

        // exportGridRagAccumulateLabels Explicit
        {
            typedef LabelsProxy<2, xt::pytensor<uint32_t, 2>> ExplicitPyLabels2D;
            typedef GridRag<2, ExplicitPyLabels2D> ExplicitLabelsGridRag2D;

            typedef LabelsProxy<3, xt::pytensor<uint32_t, 3>> ExplicitPyLabels3D;
            typedef GridRag<3, ExplicitPyLabels3D> ExplicitLabelsGridRag3D;
            // accumulate labels
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);
        }

        // exportGridRagStackedAccumulateLabels
        {
            // explicit
            {
                typedef LabelsProxy<3, xt::pytensor<uint32_t, 3>> ExplicitPyLabels3D;
                typedef GridRagStacked2D<ExplicitPyLabels3D> StackedRagUInt32;

                typedef xt::pytensor<uint32_t, 3> UInt32Array;
                typedef xt::pytensor<uint64_t, 3> UInt64Array;

                // accumulate labels
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt32, UInt32Array>(ragModule);
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt32, UInt64Array>(ragModule);
            }

            // hdf5
            #ifdef WITH_HDF5
            {
                typedef Hdf5Labels<3,uint32_t> LabelsUInt32;
                typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
                typedef Hdf5Labels<3,uint64_t> LabelsUInt64;
                typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;

                typedef nifty::hdf5::Hdf5Array<uint32_t> UInt32Array;
                typedef nifty::hdf5::Hdf5Array<uint64_t> UInt64Array;

                // accumulate labels
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt32, UInt32Array>(ragModule);
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt64, UInt32Array>(ragModule);
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt32, UInt64Array>(ragModule);
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt64, UInt64Array>(ragModule);

                // getSkipEdgesForSlice
                exportGetSkipEdgesForSliceT<StackedRagUInt32,uint32_t>(ragModule);
                //exportGetSkipEdgesForSliceT<StackedRagUInt64,uint64_t>(ragModule);
            }
            #endif
        }
    }

} // end namespace graph
} // end namespace nifty
