#ifdef WITH_FASTFILTERS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
    void exportAccumulateEdgeFeaturesFromFiltersT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeFeaturesFromFilters",
        [](
            const RAG & rag,
            DATA data,
            const int numberOfThreads
        ){

            // TODO don't hard code this
            uint64_t nChannels = 12;
            uint64_t nStats = 9;
            uint64_t nFeatures = nChannels * nStats;
            nifty::marray::PyView<float> out({uint64_t(rag.edgeIdUpperBound()+1),nFeatures});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeFeaturesFromFilters(rag, data, out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
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

            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, FloatArray>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, FloatArray>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, UInt8Array>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, UInt8Array>(ragModule);
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

            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, FloatArray>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, FloatArray>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, UInt8Array>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, UInt8Array>(ragModule);
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
#endif
