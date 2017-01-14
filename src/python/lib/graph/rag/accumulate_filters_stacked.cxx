#ifdef WITH_FASTFILTERS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/grid_rag_accumulate_filters_stacked.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif

namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;

    template<class RAG, class DATA>
    void exportAccumulateEdgeFeaturesFromFiltersInCoreT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeFeaturesFromFilters",
        [](
            const RAG & rag,
            DATA data,
            const bool keepXYOnly,
            const bool keepZOnly,
            const int numberOfThreads
        ){

            uint64_t nEdges = rag.edgeIdUpperBound()+1;
            if(keepXYOnly)
                nEdges = rag.numberOfInSliceEdges();
            else if(keepZOnly)
                nEdges = rag.numberOfInBetweenSliceEdges();
            else if(keepXYOnly && keepZOnly)
                throw std::runtime_error("keepXYOnly and keepZOnly are not allowed to be both activated!");
            // TODO don't hard code this
            uint64_t nChannels = 12;
            uint64_t nStats = 9;
            uint64_t nFeatures = nChannels * nStats;
            nifty::marray::PyView<float> out({nEdges,nFeatures});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeFeaturesFromFilters(rag, data, out, keepXYOnly, keepZOnly, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("keepXYOnly") = false,
        py::arg("keepZOnly") = false,
        py::arg("numberOfThreads")= -1
        );
    }
    
    template<class RAG, class DATA>
    void exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeFeaturesFromFilters",
        [](
            const RAG & rag,
            DATA data,
            nifty::hdf5::Hdf5Array<float> out,
            const bool keepXYOnly,
            const bool keepZOnly,
            const int numberOfThreads
        ){

            uint64_t nEdges = rag.edgeIdUpperBound()+1;
            if(keepXYOnly)
                nEdges = rag.numberOfInSliceEdges();
            else if(keepZOnly)
                nEdges = rag.numberOfInBetweenSliceEdges();
            else if(keepXYOnly && keepZOnly)
                throw std::runtime_error("keepXYOnly and keepZOnly are not allowed to be both activated!");
            // TODO don't hard code this
            uint64_t nChannels = 12;
            uint64_t nStats = 9;
            uint64_t nFeatures = nChannels * nStats;
            // need to check that this is set correct
            NIFTY_CHECK_OP(out.shape(0),==,nEdges,"Number of edges is incorrect!");
            NIFTY_CHECK_OP(out.shape(1),==,nFeatures,"Number of features is incorrect!");
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeFeaturesFromFilters(rag, data, out, keepXYOnly, keepZOnly, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("out"),
        py::arg("keepXYOnly") = false,
        py::arg("keepZOnly") = false,
        py::arg("numberOfThreads")= -1
        );
    }
    
    template<class RAG, class DATA>
    void exportAccumulateSkipEdgeFeaturesFromFiltersT(
        py::module & ragModule
    ){
        ragModule.def("accumulateSkipEdgeFeaturesFromFilters",
        [](
            const RAG & rag,
            DATA data,
            const std::vector<std::pair<size_t,size_t>> & skipEdges,
            nifty::marray::PyView<size_t> skipRanges, // TODO should these be const?
            nifty::marray::PyView<size_t> skipStarts, // TODO should these be const?
            const int numberOfThreads
        ){
            uint64_t nSkipEdges = skipEdges.size();
            std::cout << "Here, " << nSkipEdges << std::endl;
            // TODO don't hard code this
            uint64_t nChannels = 12;
            uint64_t nStats = 9;
            uint64_t nFeatures = nChannels * nStats;
            nifty::marray::PyView<float> out({nSkipEdges,nFeatures});
            {
                std::cout << "Doing stuff" << std::endl;
                py::gil_scoped_release allowThreads;
                accumulateSkipEdgeFeaturesFromFilters(rag,
                    data,
                    out,
                    skipEdges,
                    skipRanges,
                    skipStarts,
                    numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("skipEdges"),
        py::arg("skipRanges"),
        py::arg("skipStarts"),
        py::arg("numberOfThreads")= -1
        );
    }


    void exportAccumulateEdgeFeaturesFromFilters(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabels<3,uint32_t> LabelsUInt32; 
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            typedef ExplicitLabels<3,uint64_t> LabelsUInt64; 
            typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;
            typedef nifty::marray::PyView<float, 3> FloatArray;
            typedef nifty::marray::PyView<uint8_t, 3> UInt8Array;

            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, FloatArray>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt64, FloatArray>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, UInt8Array>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt64, UInt8Array>(ragModule);
        }
        
        //hdf5
        #ifdef WITH_HDF5
        {
            typedef Hdf5Labels<3,uint32_t> LabelsUInt32; 
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            typedef Hdf5Labels<3,uint64_t> LabelsUInt64; 
            typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;
            typedef nifty::hdf5::Hdf5Array<float> FloatArray;
            typedef nifty::hdf5::Hdf5Array<uint8_t> UInt8Array;

            // in core
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, FloatArray>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt64, FloatArray>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt32, UInt8Array>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersInCoreT<StackedRagUInt64, UInt8Array>(ragModule);
            //
            //// out of core
            //exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT<StackedRagUInt32, FloatArray>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT<StackedRagUInt64, FloatArray>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT<StackedRagUInt32, UInt8Array>(ragModule);
            //exportAccumulateEdgeFeaturesFromFiltersOutOfCoreT<StackedRagUInt64, UInt8Array>(ragModule);
            
            // export skipEdgeFeatures
            exportAccumulateSkipEdgeFeaturesFromFiltersT<StackedRagUInt32, FloatArray>(ragModule);
            exportAccumulateSkipEdgeFeaturesFromFiltersT<StackedRagUInt64, FloatArray>(ragModule);
            exportAccumulateSkipEdgeFeaturesFromFiltersT<StackedRagUInt32, UInt8Array>(ragModule);
            exportAccumulateSkipEdgeFeaturesFromFiltersT<StackedRagUInt64, UInt8Array>(ragModule);
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
#endif
