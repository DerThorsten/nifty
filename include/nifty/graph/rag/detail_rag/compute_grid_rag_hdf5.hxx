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


        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        Coord blockShape,blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShape[d] = std::min(settings.blockShape[d], shape[d]);
            blockShapeWithBorder[d] = std::min(blockShape[d]+1, shape[d]);
        }
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto blocksPerAxis = shape/blockShape;
        
        rag.assign(numberOfLabels);


        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        //std::cout<<"acutal n threads "<<nThreads<<"\n";
        
        auto getBlockRange = [&](const Coord & blockCoord,Coord & blockBegin,
                                Coord & blockEnd, Coord & actualBlockShape){
            for(auto d=0; d<DIM; ++d){
                blockBegin[d] = blockCoord[d] * blockShape[d];
                blockEnd[d] =   std::min(shape[d], ((blockCoord[d] + 1) * blockShape[d]) +1 );
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
            perThreadDataVec[i].blockLabels.resize(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
            perThreadDataVec[i].adjacency.resize(numberOfLabels);
        });
        
        Coord zeroCoord(0);
        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        //std::cout<<"settings.blockShape  "<<settings.blockShape<<"\n";
        //std::cout<<"blockShapeWithBorder "<<blockShapeWithBorder<<"\n";
        //std::cout<<"blocks per axis      "<<blocksPerAxis<<"\n";

        nifty::tools::parallelForEachCoordinate(threadpool, blocksPerAxis,
        [&](const int tid, const Coord & blockCoord){

            //std::cout<<"TID "<<tid<<"\n";
            // get begin end end of the block with an overlap of 1 
            Coord blockBegin, blockEnd, actualBlockShape;
            getBlockRange(blockCoord, blockBegin, blockEnd, actualBlockShape);

            auto blockLabels = perThreadDataVec[tid].blockLabels.view(zeroCoord.begin(), actualBlockShape.begin());

            Coord marrayShape;
            Coord viewShape;

            for(auto d=0; d<DIM; ++d){
                marrayShape[d] = perThreadDataVec[tid].blockLabels.shape(d);
                viewShape[d] = blockLabels.shape(d);
            }

            //std::cout<<"marrayShape      "<<marrayShape<<"\n";
            //std::cout<<"viewShape        "<<viewShape<<"\n";

            //std::cout<<"blockBegin       "<<blockBegin<<"\n";
            //std::cout<<"blockEnd         "<<blockEnd<<"\n";
            //std::cout<<"actualBlockShape "<<actualBlockShape<<"\n";

            //marray::Marray<LABEL_TYPE> buffer(actualBlockShape.begin(), actualBlockShape.end());

            // get the labels block from hdf5 
            // 
            ////std::cout<<"code readSubarray\n";
            labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);
            ////std::cout<<"done code readSubarray\n";
            ////std::cout<<"buffer "<<buffer.asString()<<"\n";

            // get the adjacency for each thread on its own
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
        });
        
        for(auto perThreadData : perThreadDataVec){
            for(const auto adjacencySet : perThreadData.adjacency){
                //std::cout<<"adj size "<<adjacencySet.size()<<"\n";
            }
        }
        rag.mergeAdjacencies(perThreadDataVec, threadpool);

    }
};



} // end namespace detail_rag
} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX */
