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
#include "nifty/tools/for_each_block.hxx"

namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class Hdf5Labels;

template<size_t DIM, class LABELS_PROXY>
class GridRag;

template<class LABEL_TYPE>
class GridRagStacked2D;


// \cond SUPPRESS_DOXYGEN
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


        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        

        const auto numberOfLabels = labelsProxy.numberOfLabels();
        rag.assign(numberOfLabels);


        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();


        // allocate / create data for each thread
        Coord blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(settings.blockShape[d]+1, shape[d]);
        }
        struct PerThreadData{
            marray::Marray<LABEL_TYPE> blockLabels;
            std::vector< container::BoostFlatSet<uint64_t> > adjacency;
        };
        std::vector<PerThreadData> perThreadDataVec(nThreads);
        parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
            perThreadDataVec[i].blockLabels.resize(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
            perThreadDataVec[i].adjacency.resize(numberOfLabels);
        });
        
        
        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };


        const Coord overlapBegin(0), overlapEnd(1);
        const Coord zeroCoord(0);

        std::mutex mtx;
        auto doneBlocks = 0;
        auto nBlocks = 0;


        
        nifty::parallel::ThreadPool stSp(nifty::parallel::ParallelOptions(1));
        tools::parallelForEachBlockWithOverlap(stSp,shape, settings.blockShape, overlapBegin, overlapEnd,
        [&](
            const int tid,
            const Coord & blockCoreBegin, const Coord & blockCoreEnd,
            const Coord & blockBegin, const Coord & blockEnd
        ){
            nBlocks += 1;
        });


        std::cout<<"nBlocks "<<nBlocks<<"\n";

        tools::parallelForEachBlockWithOverlap(threadpool,shape, settings.blockShape, overlapBegin, overlapEnd,
        [&](
            const int tid,
            const Coord & blockCoreBegin, const Coord & blockCoreEnd,
            const Coord & blockBegin, const Coord & blockEnd
        ){
            const Coord actualBlockShape = blockEnd - blockBegin;
            auto blockLabels = perThreadDataVec[tid].blockLabels.view(zeroCoord.begin(), actualBlockShape.begin());

            Coord marrayShape;
            Coord viewShape;

            for(auto d=0; d<DIM; ++d){
                marrayShape[d] = perThreadDataVec[tid].blockLabels.shape(d);
                viewShape[d] = blockLabels.shape(d);
            }

            mtx.lock();
                labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);
            mtx.unlock();

            auto & adjacency = perThreadDataVec[tid].adjacency;
            nifty::tools::forEachCoordinate(actualBlockShape,[&](const Coord & coord){
                const auto lU = blockLabels(coord.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    const auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < actualBlockShape[axis]){
                        const auto lV = blockLabels(coord2.asStdArray());
                        //std::cout<<"lU "<<lU<<" lV"<<lV<<"\n";
                        if(lU != lV){
                            //std::cout<<"HUTUHUT\n";
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });
            mtx.lock();
                doneBlocks += 1;
                std::cout<<doneBlocks<<" / "<<nBlocks<<" "<<100.0*float(doneBlocks+1)/float(nBlocks)<<"\n";
            mtx.unlock();


        });

        rag.mergeAdjacencies(perThreadDataVec, threadpool);

    }
};



} // end namespace detail_rag
// \endcond

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX */
