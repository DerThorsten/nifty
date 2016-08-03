#pragma once
#ifndef NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX
#define NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX

#include <vector>



#include "nifty/container/boost_flat_set.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class Hdf5Labels;

template<size_t DIM, class LABELS_PROXY>
class GridRag;



namespace detail_rag{

template< class GRID_RAG>
struct ComputeRag;


template<size_t DIM, class LABEL_TYPE>
struct ComputeRag< GridRag<DIM,  Hdf5Labels<DIM, LABEL_TYPE> > > {

    template<class S>
    static void computeRag(
        GridRag<DIM,  Hdf5Labels<DIM, LABEL_TYPE> > & rag,
        const S & settings
    ){
        typedef array::StaticArray<int64_t, DIM> Coord;

        const auto & blockShape = settings.blockShape;
        const auto & labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto & shape = labelsProxy.shape();
        const auto blocksPerAxis = shape/blockShape;
 


        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        
        auto getBlockRange = [&](const Coord & blockCoord,Coord & blockBegin,
                                Coord & blockEnd, Coord & actualBlockShape){
            for(auto d=0; d<DIM; ++d){
                blockBegin[d] = blockCoord[d] * blockShape[d];
                blockEnd[d] =   std::min(shape[d], (blockCoord[d] + 1 * blockShape[d]) +1 );
                actualBlockShape[d] = blockEnd[d] - blockBegin[d];
            }
        };



        // allocate / create data for each thread
        struct PerThreadData{
            marray::Marray<LABEL_TYPE> blockLabels;
            std::vector< container::BoostFlatSet<uint64_t> > adjacency;
        };
        std::vector<PerThreadData> perThreadDataVec(nThreads);
        parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
            perThreadDataVec[i].blockLabels.resize(blockShape.begin(), blockShape.end());
            perThreadDataVec[i].adjacency.resize(numberOfLabels);
        });
        
        Coord zeroCoord(0);
        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        nifty::tools::parallelForEachCoordinate(threadpool, blocksPerAxis,
        [&](const int tid, const Coord & blockCoord){

            // get begin end end of the block with an overlap of 1 
            Coord blockBegin, blockEnd, actualBlockShape;
            getBlockRange(blockCoord, blockBegin, blockEnd, actualBlockShape);

            // get the labels buffer
            auto blockLabels = perThreadDataVec[tid].blockLabels.view(zeroCoord.begin(), actualBlockShape.begin());

            // get the labels block from hdf5 
            labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);

            // get the adjacency for each thread on its own
            auto & adjacency = perThreadDataVec[tid].adjacency;

            nifty::tools::forEachCoordinate(actualBlockShape,[&](const Coord & coord){
                const auto lU = blockLabels(coord.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    const auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < shape[axis]){
                        const auto lV = blockLabels(coord2.asStdArray());
                        if(lU != lV){
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });
        });
        
        rag.mergeAdjacencies(perThreadDataVec, threadpool);

    }
};



} // end namespace detail_rag
} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX */
