#pragma once
#ifndef NIFTY_TOOLS_FOR_EACH_BLOCK_HXX
#define NIFTY_TOOLS_FOR_EACH_BLOCK_HXX

#include <sstream>
#include <chrono>
#include <array>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace tools{

    
    
    template<size_t DIM, class SHAPE_T, class BLOCK_SHAPE_T, class F>
    void parallelForEachBlock(
        parallel::ThreadPool & threadpool,
        const array::StaticArray<SHAPE_T, DIM> & shape,
        const array::StaticArray<BLOCK_SHAPE_T, DIM> & blockShape,
        F && f
    ){
        typedef array::StaticArray<int64_t, DIM> Coord;
        Coord blocksPerAxis, actualblocksShape;

        for(auto d=0; d<DIM; ++d){
            actualblocksShape[d] = std::min(int64_t(blockShape[d]), int64_t(shape[d]));
            blocksPerAxis[d] = shape[d] / actualblocksShape[d];
        }

        parallelForEachCoordinate(threadpool, blocksPerAxis,
        [&](const int tid, const Coord & blockCoord){
            Coord blockBegin, blockEnd;
            for(auto d=0; d<DIM; ++d){
                blockBegin[d] = blockCoord[d] * actualblocksShape[d];
                blockEnd[d] =  std::min(shape[d], (blockCoord[d] + 1) * blockShape[d]);
            }
            f(tid, blockBegin, blockEnd);
        });
    }


    template<size_t DIM, class SHAPE_T, class BLOCK_SHAPE_T, class OVERLAP_SHAPE_T, class F>
    void parallelForEachBlockWithOverlap(
        parallel::ThreadPool & threadpool,
        const array::StaticArray<SHAPE_T, DIM> &    shape,
        const array::StaticArray<BLOCK_SHAPE_T, DIM> & blockShape,
        const array::StaticArray<OVERLAP_SHAPE_T, DIM> & overlapBegin,
        const array::StaticArray<OVERLAP_SHAPE_T, DIM> & overlapEnd,
        F && f
    ){

        typedef array::StaticArray<int64_t, DIM> Coord;
        Coord blocksPerAxis, actualblocksShape;

        for(auto d=0; d<DIM; ++d){
            actualblocksShape[d] = std::min(int64_t(blockShape[d]), int64_t(shape[d]));
            blocksPerAxis[d] = shape[d] / actualblocksShape[d];
        }

        parallelForEachCoordinate(threadpool, blocksPerAxis,
        [&](const int tid, const Coord & blockCoord){

            Coord blockBegin, blockEnd, blockWithOlBegin, blockWithOlEnd;

            for(auto d=0; d<DIM; ++d){
                const int64_t bc = blockCoord[d];
                const int64_t bs = actualblocksShape[d];

                blockBegin[d] = blockCoord[d] * bs;
                blockEnd[d] =  std::min(shape[d], (blockCoord[d] + 1) * bs);

                const int64_t bBegin = blockBegin[d]  - int64_t(overlapBegin[d]);
                const int64_t bEnd =   blockEnd[d] + int64_t(overlapEnd[d]);

                blockWithOlBegin[d] =  std::max(int64_t(0), bBegin);
                blockWithOlEnd[d] =  std::min(shape[d], bEnd);
            }
            f(tid, blockBegin, blockEnd, blockWithOlBegin, blockWithOlEnd);
        });
    }




} // end namespace nifty::tools
} // end namespace nifty

#endif /*NIFTY_TOOLS_FOR_EACH_BLOCK_HXX*/
