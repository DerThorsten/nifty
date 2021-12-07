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
                                // Get the offset vector:
                                typedef typename std::vector<array::StaticArray<int64_t, DIM>> OffsetVectorType;
                                // OffsetVectorType offsetVector(offsets.shape()[0]);
                                OffsetVectorType offsetVector(offsets.size());
                                for(auto i=0; i<offsetVector.size(); ++i){
                                    for(auto d=0; d<DIM; ++d){
                                        offsetVector[i][d] = offsets[i][d];
                                        // offsetVector[i][d] = offsets(i,d);
                                    }
                                }

                                // Initialize vector that will contain the longRange uvIds:
                                nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                                nifty::parallel::ThreadPool threadpool(pOpts);
                                const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();
                                std::vector<std::vector<std::pair<uint64_t,uint64_t>>> longRangePairs(actualNumberOfThreads);

                                // Look for lifted edges:
                                {
                                    py::gil_scoped_release allowThreads;
                                    computeLiftedEdgesFromRagAndOffsets(rag, offsetVector, longRangePairs, numberOfThreads);
                                }

                                // Get the total number of found edges:
                                int total_nb_edges = 0;
                                std::vector<int> index_offset(actualNumberOfThreads);
                                for (int i=0; i<actualNumberOfThreads; i++) {
                                    index_offset[i] = total_nb_edges;
                                    total_nb_edges += (int) longRangePairs[i].size();
                                }

                                // Create output tensor:
                                xt::pytensor<uint64_t, 2> out_tensor = xt::zeros<float>({(std::size_t) total_nb_edges, (std::size_t) 2});

                                // Collect results in output tensor:
                                for (int i=0; i<actualNumberOfThreads; i++) {
                                    for (int j=0; j<longRangePairs[i].size(); j++) {
                                        out_tensor(index_offset[i]+j,0) = longRangePairs[i][j].first;
                                        out_tensor(index_offset[i]+j,1) = longRangePairs[i][j].second;
                                    }
                                }

                                return out_tensor;

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


