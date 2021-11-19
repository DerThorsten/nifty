#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include <cstddef>
#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/get_lifted_edges_from_rag_and_offsets.hxx"



namespace py = pybind11;


namespace nifty {
    namespace graph {

        using namespace py;

        template<std::size_t DIM, class RAG>
        void exportComputeLiftedEdgesFromRagAndOffsets(
                py::module & ragModule
        ){
            ragModule.def("computeLiftedEdgesFromRagAndOffsets_impl",
                            [](
                                    const RAG & rag,
                                    const std::vector<std::vector<int>> & offsets,
                                    // xt::pytensor<int64_t, 2> offsets,
                                    const int numberOfThreads
                            ){
                                typedef typename std::vector<array::StaticArray<int64_t, DIM>> OffsetVectorType;
                                // OffsetVectorType offsetVector(offsets.shape()[0]);
                                OffsetVectorType offsetVector(offsets.size());
                                for(auto i=0; i<offsetVector.size(); ++i){
                                    for(auto d=0; d<DIM; ++d){
                                        offsetVector[i][d] = offsets[i][d];
                                        // offsetVector[i][d] = offsets(i,d);
                                    }
                                }

                                return computeLiftedEdgesFromRagAndOffsets(rag, offsetVector, numberOfThreads);

                            },
                            py::arg("rag"),
                            py::arg("offsets"),
                            py::arg("numberOfThreads") = -1
            );
        };

        void exportComputeLiftedEdges(py::module & ragModule) {

            //explicit
            {

                typedef xt::pytensor<uint32_t, 2> ExplicitLabels2D;
                typedef GridRag<2, ExplicitLabels2D> Rag2d;
                typedef xt::pytensor<uint32_t, 3> ExplicitLabels3D;
                typedef GridRag<3, ExplicitLabels3D> Rag3d;

                exportComputeLiftedEdgesFromRagAndOffsets<2, Rag2d>(ragModule);
                exportComputeLiftedEdgesFromRagAndOffsets<3, Rag3d>(ragModule);

            }
        }

    }
}


