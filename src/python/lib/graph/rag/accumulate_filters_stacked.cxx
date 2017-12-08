#ifdef WITH_FASTFILTERS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/graph/rag/grid_rag_labels_proxy.hxx"
#include "nifty/graph/rag/feature_accumulation/grid_rag_accumulate_filters_stacked.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;

    // TODO assert that z direction is in (0,1,2)
    template<class RAG, class DATA>
    void exportAccumulateEdgeFeaturesFromFiltersInCoreT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeFeaturesFromFilters",
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

            // TODO don't hard code this
            uint64_t nChannels = 12;
            uint64_t nStats = 9;
            uint64_t nFeatures = nChannels * nStats;
            nifty::marray::PyView<float> outXY({nEdgesXY,nFeatures});
            nifty::marray::PyView<float> outZ({nEdgesZ,nFeatures});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeFeaturesFromFilters(rag, data, outXY, outZ, keepXYOnly, keepZOnly, zDirection, numberOfThreads);
            }
            return std::make_tuple(outXY, outZ);
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("keepXYOnly") = false,
        py::arg("keepZOnly") = false,
        py::arg("zDirection") = 0,
        py::arg("numberOfThreads")= -1
        );
    }

    // TODO assert that z direction is in (0,1,2)
    template<class RAG, class DATA>
    void exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeFeaturesFromFilters",
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

            // TODO don't hard code this
            uint64_t nChannels = 12;
            uint64_t nStats = 9;
            uint64_t nFeatures = nChannels * nStats;
            // need to check that this is set correct
            NIFTY_CHECK_OP(outXY.shape(0),==,nEdgesXY,"Number of edges is incorrect!");
            NIFTY_CHECK_OP(outZ.shape(0),==,nEdgesZ,"Number of edges is incorrect!");
            NIFTY_CHECK_OP(outXY.shape(1),==,nFeatures,"Number of features is incorrect!");
            NIFTY_CHECK_OP(outZ.shape(1),==,nFeatures,"Number of features is incorrect!");
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeFeaturesFromFilters(rag, data, outXY, outZ, keepXYOnly, keepZOnly, zDirection, numberOfThreads);
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

    // TODO assert that z direction is in (0,1,2)
    template<class RAG, class DATA>
    void exportAccumulateSkipEdgeFeaturesFromFiltersT(
        py::module & ragModule
    ){
        ragModule.def("accumulateSkipEdgeFeaturesFromFilters",
        [](
            const RAG & rag,
            DATA & data,
            const std::vector<std::pair<size_t,size_t>> & skipEdges,
            const std::vector<size_t> & skipRanges,
            const std::vector<size_t> & skipStarts,
            const int zDirection,
            const int numberOfThreads
        ){
            uint64_t nSkipEdges = skipEdges.size();

            // TODO don't hard code this
            uint64_t nChannels = 12;
            uint64_t nStats = 9;
            uint64_t nFeatures = nChannels * nStats;
            nifty::marray::PyView<float> out({nSkipEdges,nFeatures});
            {
                py::gil_scoped_release allowThreads;
                accumulateSkipEdgeFeaturesFromFilters(rag,
                    data,
                    out,
                    skipEdges,
                    skipRanges,
                    skipStarts,
                    zDirection,
                    numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("skipEdges"),
        py::arg("skipRanges"),
        py::arg("skipStarts"),
        py::arg("zDirection") = 0,
        py::arg("numberOfThreads")= -1
        );
    }


    void exportAccumulateEdgeFeaturesFromFilters(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabels<3,uint32_t> LabelsUInt32;
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            typedef nifty::marray::PyView<float, 3> FloatArray;
            typedef nifty::marray::PyView<uint8_t, 3> UInt8Array;

            exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, FloatArray>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, UInt8Array>(ragModule);
        }

        //hdf5
        #ifdef WITH_HDF5
        {
            typedef Hdf5Labels<3,uint32_t> LabelsUInt32;
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            typedef nifty::hdf5::Hdf5Array<float> FloatArray;
            typedef nifty::hdf5::Hdf5Array<uint8_t> UInt8Array;

            // in core
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, FloatArray>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, UInt8Array>(ragModule);

            // out of core
            exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT<StackedRagUInt32, FloatArray>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT<StackedRagUInt32, UInt8Array>(ragModule);

            // export skipEdgeFeatures
            exportAccumulateSkipEdgeFeaturesFromFiltersT<StackedRagUInt32, FloatArray>(ragModule);
            exportAccumulateSkipEdgeFeaturesFromFiltersT<StackedRagUInt32, UInt8Array>(ragModule);
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
#endif
