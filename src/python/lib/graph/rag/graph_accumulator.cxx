#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag_features.hxx"






namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;


    template<class RAG,class T,std::size_t DATA_DIM>
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




    void exportGraphAccumulator(py::module & ragModule) {

        // exportGridRagAccumulateLabels
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
            typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;
            // accumulate labels
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);

            // accumulate labels
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);


        }
    }

} // end namespace graph
} // end namespace nifty
