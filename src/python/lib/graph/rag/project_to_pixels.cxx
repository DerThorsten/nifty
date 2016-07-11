#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../converter.hxx"

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



    void exportProjectToPixels(py::module & ragModule) {


        typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
        typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;


        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint64_t, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint64_t, 3>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, float, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, float, 3>(ragModule);
    }

} // end namespace graph
} // end namespace nifty
    
