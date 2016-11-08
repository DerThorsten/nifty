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

    template<class RAG, class DATA_T>
    void exportAccumulateEdgeFeaturesFromFiltersT(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeFeaturesFromFilters",
        [](
            const RAG & rag,
            nifty::marray::PyView<DATA_T, 3> data,
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

            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, float>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, uint8_t>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, float>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, uint8_t>(ragModule);
        }
        
        //hdf5
        #ifdef WITH_HDF5
        {
            typedef Hdf5Labels<3,uint32_t> LabelsUInt32; 
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            typedef Hdf5Labels<3,uint64_t> LabelsUInt64; 
            typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;

            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, float>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt32, uint8_t>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, float>(ragModule);
            exportAccumulateEdgeFeaturesFromFiltersT<StackedRagUInt64, uint8_t>(ragModule);
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
