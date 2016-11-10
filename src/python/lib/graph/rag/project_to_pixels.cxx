#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/project_to_pixels.hxx"
#include "nifty/graph/rag/project_to_pixels_stacked.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif


namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class RAG,class T,size_t DATA_DIM, bool AUTO_CONVERT>
    void exportProjectScalarNodeDataToPixelsT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
               const RAG & rag,
                nifty::marray::PyView<T, 1, AUTO_CONVERT> nodeData,
               const int numberOfThreads
           ){  
                const auto labelsProxy = rag.labelsProxy();
                const auto & shape = labelsProxy.shape();
                const auto labels = labelsProxy.labels(); 

                nifty::marray::PyView<T, DATA_DIM> pixelData(shape.begin(),shape.end());
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
                return pixelData;
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("numberOfThreads")=-1
        );
    }
    
    
    template<class RAG, class T, bool AUTO_CONVERT>
    void exportProjectScalarNodeDataToPixelsStackedExplicitT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const RAG & rag,
                nifty::marray::PyView<T, 1, AUTO_CONVERT> nodeData,
                const int numberOfThreads
           ){  
                const auto & shape = rag.shape();

                nifty::marray::PyView<T,3> pixelData(shape.begin(),shape.end());
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
    template<class RAG, class T, bool AUTO_CONVERT>
    void exportProjectScalarNodeDataToPixelsStackedHdf5T(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const RAG & rag,
                nifty::marray::PyView<T, 1, AUTO_CONVERT> nodeData,
                nifty::hdf5::Hdf5Array<T> pixelData,
                const int numberOfThreads
           ){  
                const auto & shape = rag.shape();

                for(int d = 0; d < 3; ++d)
                    NIFTY_CHECK_OP(shape[d],==,pixelData.shape(d),"OutShape and Rag shape do not match!")

                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
                return pixelData;
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("pixelData"),py::arg("numberOfThreads")=-1
        );
    }
    #endif


    void exportProjectToPixels(py::module & ragModule) {

        // exportScalarNodeDataToPixels
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
            typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;

            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint32_t, 2, false>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint32_t, 3, false>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint64_t, 2, false>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint64_t, 3, false>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, float, 2, false>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, float, 3, false>(ragModule);

            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, double, 2, true>(ragModule);
            exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, double, 3, true>(ragModule);
        }
        
        // exportScalarNodeDataToPixelsStacked
        {
            // explicit
            {
                typedef ExplicitLabels<3,uint32_t> LabelsUInt32; 
                typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
                typedef ExplicitLabels<3,uint64_t> LabelsUInt64; 
                typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;

                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt32,uint32_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt32,uint64_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt32,float,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt32,double,true>(ragModule);
                
                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt64,uint32_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt64,uint64_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt64,float,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedExplicitT<StackedRagUInt64,double,true>(ragModule);
            
            }
            
            // hdf5 
            #ifdef WITH_HDF5
            {
                typedef Hdf5Labels<3,uint32_t> LabelsUInt32; 
                typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
                typedef Hdf5Labels<3,uint64_t> LabelsUInt64; 
                typedef GridRagStacked2D<LabelsUInt64> StackedRagUInt64;
                
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt32,uint32_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt32,uint64_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt32,float,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt32,double,true>(ragModule);
                
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt64,uint32_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt64,uint64_t,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt64,float,false>(ragModule);
                exportProjectScalarNodeDataToPixelsStackedHdf5T<StackedRagUInt64,double,true>(ragModule);
            
            }
            #endif

        }
    }

} // end namespace graph
} // end namespace nifty
    
