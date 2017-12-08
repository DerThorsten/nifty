#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/feature_accumulation/grid_rag_accumulate_stacked.hxx"

namespace py = pybind11;

namespace nifty{
namespace graph{

    using namespace py;

    template<class RAG, class DATA>
    void exportAccumulateEdgeStandardFeaturesStackedInCoreT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStandardFeatures",
        [](
            const RAG & rag,
            DATA & data,
            const bool keepXYOnly,
            const bool keepZOnly,
            const int zDirection,
            const int numberOfThreads
        ){
            if(keepXYOnly && keepZOnly)
                throw std::runtime_error("keepXYOnly and keepZOnly are not allowed to be both activated!");
            uint64_t nEdgesXY = !keepZOnly ? rag.numberOfInSliceEdges() : 1L;
            uint64_t nEdgesZ  = !keepXYOnly ? rag.numberOfInBetweenSliceEdges() : 1L;
            uint64_t nStats = 9;

            nifty::marray::PyView<float> outXY({nEdgesXY, nStats});
            nifty::marray::PyView<float> outZ({nEdgesZ, nStats});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandardFeatures(rag, data, outXY, outZ, keepXYOnly, keepZOnly, zDirection, numberOfThreads);
            }
            return std::make_tuple(outXY, outZ);
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("keepXYOnly") = false,
        py::arg("keepZOnly") = false,
        py::arg("zDirection")= 0,
        py::arg("numberOfThreads")= -1
        );
    }


    #ifdef WITH_HDF5
    template<class RAG, class DATA>
    void exportAccumulateEdgeStandardFeaturesOutOfCoreT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStandardFeatures",
        [](
            const RAG & rag,
            DATA & data,
            nifty::hdf5::Hdf5Array<float> & outXY,
            nifty::hdf5::Hdf5Array<float> & outZ,
            const bool keepXYOnly,
            const bool keepZOnly,
            const int zDirection,
            const int numberOfThreads
        ){

            if(keepXYOnly && keepZOnly)
                throw std::runtime_error("keepXYOnly and keepZOnly are not allowed to be both activated!");
            uint64_t nEdgesXY = !keepZOnly ? rag.numberOfInSliceEdges() : 1L;
            uint64_t nEdgesZ  = !keepXYOnly ? rag.numberOfInBetweenSliceEdges() : 1L;

            uint64_t nFeatures = 9;
            // need to check that this is set correct
            NIFTY_CHECK_OP(outXY.shape(0),==,nEdgesXY,"Number of edges is incorrect!");
            NIFTY_CHECK_OP(outZ.shape(0),==,nEdgesZ,"Number of edges is incorrect!");
            NIFTY_CHECK_OP(outXY.shape(1),==,nFeatures,"Number of features is incorrect!");
            NIFTY_CHECK_OP(outZ.shape(1),==,nFeatures,"Number of features is incorrect!");
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandardFeatures(rag, data, outXY, outZ, keepXYOnly, keepZOnly, zDirection, numberOfThreads);
            }
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("outXY"),
        py::arg("outZ"),
        py::arg("keepXYOnly") = false,
        py::arg("keepZOnly") = false,
        py::arg("zDirection") = 0,
        py::arg("numberOfThreads")= -1
        );
    }
    #endif


    template<class RAG>
    void exportGetSkipEdgeLengthsT(
        py::module & ragModule
    ){
        ragModule.def("getSkipEdgeLengths",
        [](
            const RAG & rag,
            const std::vector<std::pair<size_t,size_t>> & skipEdges,
            const std::vector<size_t> & skipRanges,
            const std::vector<size_t> & skipStarts,
            const int numberOfThreads
        ){
            size_t nSkipEdges = skipEdges.size();
            std::vector<size_t> out(nSkipEdges);
            {
                py::gil_scoped_release allowThreads;
                getSkipEdgeLengths(rag,
                    out,
                    skipEdges,
                    skipRanges,
                    skipStarts,
                    numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("skipEdges"),
        py::arg("skipRanges"),
        py::arg("skipStarts"),
        py::arg("numberOfThreads")= -1
        );
    }


    void exportAccumulateStacked(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabels<3,uint32_t> LabelsUInt32;
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            typedef ExplicitLabels<3,uint64_t> LabelsUInt64;
            typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;
            typedef nifty::marray::PyView<float, 3> FloatArray;
            typedef nifty::marray::PyView<uint8_t, 3> UInt8Array;

            exportAccumulateEdgeStandardFeaturesStackedInCoreT<StackedRagUInt32,FloatArray>(ragModule);
            exportAccumulateEdgeStandardFeaturesStackedInCoreT<StackedRagUInt32,UInt8Array>(ragModule);
        }
        // hdf5
        #ifdef WITH_HDF5
        {
            typedef Hdf5Labels<3,uint32_t> LabelsUInt32;
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            typedef Hdf5Labels<3,uint64_t> LabelsUInt64;
            typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;
            typedef nifty::hdf5::Hdf5Array<float> FloatArray;
            typedef nifty::hdf5::Hdf5Array<uint8_t> UInt8Array;

            // out of core
            exportAccumulateEdgeStandardFeaturesOutOfCoreT<StackedRagUInt32, FloatArray>(ragModule);
            exportAccumulateEdgeStandardFeaturesOutOfCoreT<StackedRagUInt64, FloatArray>(ragModule);
            //exportAccumulateEdgeStandardFeaturesOutOfCoreT<StackedRagUInt32, UInt8Array>(ragModule);
            //exportAccumulateEdgeStandardFeaturesOutOfCoreT<StackedRagUInt64, UInt8Array>(ragModule);

            exportGetSkipEdgeLengthsT<StackedRagUInt32>(ragModule);
            exportGetSkipEdgeLengthsT<StackedRagUInt64>(ragModule);
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
