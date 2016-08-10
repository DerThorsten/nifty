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
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){

            nifty::marray::PyView<DATA_T> out({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(2)});
            array::StaticArray<int64_t, DIM> blocKShape_;
            accumulateEdgeMeanAndLength(rag, data, blocKShape, out, numberOfThreads);
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }


    template<size_t DIM, class RAG, class DATA_T>
    void exportAccumulateMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateMeanAndLength",
        [](
            const RAG & rag,
            nifty::marray::PyView<DATA_T, DIM> data,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;

            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(2)});
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(2)});
            array::StaticArray<int64_t, DIM> blocKShape_;
            accumulateMeanAndLength(rag, data, blocKShape, edgeOut, nodeOut, numberOfThreads);

            return OutType(edgeOut, nodeOut);;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }



    void exportAccumulate(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> Rag2d;
            typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d;

            exportAccumulateEdgeMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateEdgeMeanAndLength<3, Rag3d, float>(ragModule);
            exportAccumulateMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateMeanAndLength<3, Rag3d, float>(ragModule);
        }
    }

} // end namespace graph
} // end namespace nifty
    
