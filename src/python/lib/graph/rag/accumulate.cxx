#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<size_t DIM, class RAG, class DATA_T>
    void exportAccumulateEdgeMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeMeanAndLength",
        [](
            const RAG & rag,
            nifty::marray::PyView<DATA_T, DIM> data,
            const int numberOfThreads
        ){

            nifty::marray::PyView<DATA_T> out({rag.edgeIdUpperBound()});
            accumulateEdgeMeanAndLength(rag, data, out, numberOfThreads);

        });
    }


    void exportAccumulate(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> Rag2d;
            typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d;

            exportAccumulateEdgeMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateEdgeMeanAndLength<3, Rag3d, float>(ragModule);
        }
    }

} // end namespace graph
} // end namespace nifty
    
