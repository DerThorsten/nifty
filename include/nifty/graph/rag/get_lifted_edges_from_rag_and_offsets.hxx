#pragma once

#include <random>
#include <thread>
#include "xtensor/xexpression.hpp"

#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "nifty/graph/undirected_list_graph.hxx"

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include <cstdlib>

#include <pybind11/pybind11.h>
namespace py = pybind11;


namespace nifty{
    namespace graph{

        template<std::size_t DIM, class RAG>
        auto computeLiftedEdgesFromRagAndOffsets(
                const RAG & rag,
                const std::vector<array::StaticArray<int64_t, DIM>> & offsets,
                // OUTPUT & outTensor,
                const int numberOfThreads
        ) {
            // Check whether some of the offsets
            const auto & labels = rag.labels();
            const auto & shape = rag.shape();

            nifty::parallel::ParallelOptions pOpts(numberOfThreads);
            nifty::parallel::ThreadPool threadpool(pOpts);
            const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();
            std::vector<std::vector<std::pair<uint64_t,uint64_t>>> longRangePairs(actualNumberOfThreads);


            // Look for lifted edges:
            {
                py::gil_scoped_release allowThreads;
                nifty::tools::parallelForEachCoordinate(threadpool,
                            shape,
                    [&](const auto threadId, const auto & coordP){
                        const auto u = labels[coordP];
                        for(int io=0; io<offsets.size(); ++io){
                            const auto offset = offsets[io];
                            const auto coordQ = offset + coordP;
                            // Check if both coordinates are in the volume:
                            if(coordQ.allInsideShape(shape)){
                                const auto v = labels[coordQ];

                                // Insert new edge in graph:
                                if (u != v) {
                                    const auto edge = rag.findEdge(u, v);
                                    if (edge < 0) {
                                        const std::pair <uint64_t, uint64_t> newPair(u, v);
                                        longRangePairs[threadId].push_back(newPair);
                                    }
                                }
                            }
                        }
                    }
                    );

            }

            int total_nb_edges = 0;
            std::vector<int> index_offset(actualNumberOfThreads);
            for (int i=0; i<actualNumberOfThreads; i++) {
                index_offset[i] = total_nb_edges;
                total_nb_edges += longRangePairs[i].size();
            }

            // Create output tensor:
            typename xt::xtensor<uint64_t, 2>::shape_type retshape;
            retshape[0] = total_nb_edges;
            retshape[1] = 2;
            xt::xtensor<uint64_t, 2> out(retshape);

            // Collect results:
            {
                py::gil_scoped_release allowThreads;
                for (int i=0; i<actualNumberOfThreads; i++) {
                    for (int j=0; j<longRangePairs[i].size(); j++) {
                        out(index_offset[i]+j,0) = longRangePairs[i][j].first;
                        out(index_offset[i]+j,1) = longRangePairs[i][j].second;
                    }
                }
            }
            return(out);

        }


    }
}

