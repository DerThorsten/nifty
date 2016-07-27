#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/project_to_pixels.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class RAG,class T,size_t DATA_DIM>
    void exportProjectScalarNodeDataToPixelsT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
               const RAG & rag,
                nifty::marray::PyView<T, 1> nodeData,
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
           py::arg("graph"),py::arg("nodeData"),py::arg("numberOfThreads")
        );
    }
    
    template<class RAG,class T>
    void exportProjectScalarNodeDataToPixelsSlicedT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixelsSliced",
           [](
                const RAG & rag,
                nifty::marray::PyView<T, 1> nodeData,
                const std::string & outputFile,
                const std::string & key,
                const int numberOfThreads
           ){  
                const auto labelsProxy = rag.labelsProxy();
                const auto & labels = labelsProxy.labels(); 
                const auto shape = labels.shape();

                vigra::HDF5File h5_file(outputFile, vigra::HDF5File::ReadWrite);
                typename vigra::ChunkedArrayHDF5<3, T>::shape_type chunk_shape(labels.file_.getChunkShape(key).begin());
                vigra::ChunkedArrayHDF5<3, T> pixelData(h5_file, key, 
                    vigra::HDF5File::ReadWrite, shape, chunk_shape);
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
                // For now, we don't return it, because there are no proper pythonbindings for the chunked array yet
                //return pixelData;
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("outputFile"),py::arg("key"),py::arg("numberOfThreads")
        );
    }



    void exportProjectToPixels(py::module & ragModule) {


        typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
        typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;
        
        typedef ChunkedLabelsGridRagSliced<uint32_t> ChunkedLabelsGridRag;


        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);
        exportProjectScalarNodeDataToPixelsSlicedT<ChunkedLabelsGridRag, uint32_t>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint64_t, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint64_t, 3>(ragModule);
        exportProjectScalarNodeDataToPixelsSlicedT<ChunkedLabelsGridRag, uint64_t>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, float, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, float, 3>(ragModule);
        exportProjectScalarNodeDataToPixelsSlicedT<ChunkedLabelsGridRag, float>(ragModule);
    }

} // end namespace graph
} // end namespace nifty
    
