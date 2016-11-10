#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag_features.hxx"
#include "nifty/graph/rag/grid_rag_features_stacked.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif



namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;


    template<class RAG,class T,size_t DATA_DIM>
    void exportGridRagAccumulateLabelsT(py::module & ragModule){

        ragModule.def("gridRagAccumulateLabels",
            [](
                const RAG & rag,
                nifty::marray::PyView<T, DATA_DIM> labels
            ){  
                nifty::marray::PyView<T> nodeLabels({rag.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateLabels(rag, labels, nodeLabels);
                }
                return nodeLabels;

            },
            py::arg("graph"),py::arg("labels")
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
                typedef typename DATA::DataType DataType;
                nifty::marray::PyView<DataType> nodeLabels({rag.numberOfNodes()});
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




    void exportGraphAccumulator(py::module & ragModule) {

        // exportGridRagAccumulateLabels Explicit
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
            typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;
            // accumulate labels
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);
        }
        
        // exportGridRagStackedAccumulateLabels
        {
            // explicit
            {
                typedef ExplicitLabels<3,uint32_t> LabelsUInt32; 
                typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
                typedef ExplicitLabels<3,uint64_t> LabelsUInt64; 
                typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;
            
                typedef nifty::marray::PyView<uint32_t, 3> UInt32Array;
                typedef nifty::marray::PyView<uint64_t, 3> UInt64Array;
                
                // accumulate labels
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt32, UInt32Array>(ragModule);
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt64, UInt32Array>(ragModule);
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt32, UInt64Array>(ragModule);
                exportGridRagStackedAccumulateLabelsT<StackedRagUInt64, UInt64Array>(ragModule);
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
            }
            #endif

        }
    }

} // end namespace graph
} // end namespace nifty
    
