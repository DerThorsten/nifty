#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag_accumulate_stacked.hxx"

namespace py = pybind11;

namespace nifty{
namespace graph{

    using namespace py;

    template<class RAG, class DATA>
    void exportAccumulateEdgeStandartFeaturesStacked(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStandartFeatures",
        [](
            const RAG & rag,
            DATA data,
            const double minVal,
            const double maxVal,
            const int numberOfThreads
        ){
            typedef typename DATA::DataType DataType;
            typedef nifty::marray::PyView<DataType> NumpyArrayType;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeatures(rag, data, minVal, maxVal, edgeOut, numberOfThreads);
            }
            return edgeOut;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("minVal"),
        py::arg("maxVal"),
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

            exportAccumulateEdgeStandartFeaturesStacked<StackedRagUInt32,FloatArray>(ragModule);
            exportAccumulateEdgeStandartFeaturesStacked<StackedRagUInt32,UInt8Array>(ragModule);
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
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
