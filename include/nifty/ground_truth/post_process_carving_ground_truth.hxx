#pragma once
#ifndef NIFTY_GROUND_TRUTH_POST_PROCESS_CARVING_GROUND_TRUTH_HXX
#define NIFTY_GROUND_TRUTH_POST_PROCESS_CARVING_GROUND_TRUTH_HXX

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/region_growing/seeded_region_growing.hxx"
#include "nifty/tools/timer.hxx"


namespace nifty{
namespace ground_truth{


    template<   
        size_t DIM, 
        class GROW_MAP_PIXEL_TYPE, 
        class GROUND_TRUTH_PIXEL_TYPE, 
        class P_GROUND_TRUTH_PIXEL_TYPE
    >
    void postProcessCarvingNeuroGroundTruth(
        const marray::View<GROW_MAP_PIXEL_TYPE>   & growMap,
        const marray::View<GROUND_TRUTH_PIXEL_TYPE> & groundTruth,
        marray::View<P_GROUND_TRUTH_PIXEL_TYPE> & processedGroundTruth,
        int shrinkSizeObjects = 2,
        int shrinkSizeBg = 3,
        const uint16_t numberOfQueues = 256,
        const int numberOfThreads = -1,
        const int verbose = 1
    ){
        typedef array::StaticArray<int64_t, DIM> CoordType;

        parallel::ParallelOptions pOpts(numberOfThreads);
        parallel::ThreadPool  threadpool(pOpts);

        CoordType shape;
        for(size_t i=0; i<DIM; ++i){
            shape[i] = growMap.shape(i);
        }   

        // get the grow map
        marray::Marray<uint16_t> integralGrowMap(growMap.shapeBegin(), growMap.shapeEnd());


        if(verbose)
            std::cout<<"prepare maps\n";
        // fill out with initial segmentation value + 1
        // and prepare integral grow map
        tools::parallelForEachCoordinate(threadpool, shape,
        [&](const int tid, const CoordType & coord){
            processedGroundTruth(coord.asStdArray()) = groundTruth(coord.asStdArray()) + 1;
            integralGrowMap(coord.asStdArray()) = static_cast<uint16_t>( float(numberOfQueues)*growMap(coord.asStdArray()));
        });

        
        if(verbose)
            std::cout<<"shrink iter 0 \n";
        tools::parallelForEachCoordinate(threadpool, shape,
        [&](const int tid, const CoordType & coord){

            const auto lU = groundTruth(coord.asStdArray());

                
            for(uint8_t d=0; d<DIM; ++d){

                if(coord[d]+1 < shape[d]){
                    CoordType coord2 = coord;
                    coord2[d] = coord[d]+1;
                    const auto lV = groundTruth(coord2.asStdArray());

                    // mark as "unseeded"
                    if(lU != lV){
                        processedGroundTruth(coord.asStdArray()) = 0;
                        processedGroundTruth(coord2.asStdArray()) = 0;
                    }
                }
            }

        });

        --shrinkSizeObjects;
        --shrinkSizeBg;

        const auto maxShrinkSize = std::max(shrinkSizeBg, shrinkSizeObjects);

        marray::Marray<int64_t> buffer(processedGroundTruth);

        std::cout<<"shrinkSizeObjects "<<shrinkSizeObjects<<"\n";
        std::cout<<"shrinkSizeBg "<<shrinkSizeBg<<"\n";
        std::cout<<"maxShrinkSize "<<maxShrinkSize<<"\n";


        int64_t val =  0;
        //int64_t val = -1;
        for(auto shrinkIteration=0; shrinkIteration<maxShrinkSize; ++ shrinkIteration){
             if(verbose)
                std::cout<<"shrink iter "<<shrinkIteration+1<<" \n";

            tools::parallelForEachCoordinate(threadpool, shape,
            [&](const int tid, const CoordType & coord){

                const auto lU = buffer(coord.asStdArray());

                    
                for(uint8_t d=0; d<DIM; ++d){

                    if(coord[d]+1 < shape[d]){
                        CoordType coord2 = coord;
                        coord2[d] = coord[d]+1;
                        const auto lV = buffer(coord2.asStdArray());

                        // mark as "unseeded"
                        if(lU != lV && (lU==val || lV==val)){
                                
                            //NIFTY_CHECK(lU==0 || lV==0, "internal error");

                            if(lU==1 || lV==1){
                                if(shrinkIteration < shrinkSizeObjects){
                                    if(lU == 1)
                                        buffer(coord.asStdArray()) = val-1;
                                    else
                                        buffer(coord2.asStdArray()) = val-1;
                                }
                            }
                            else{
                                if(shrinkIteration < shrinkSizeObjects){
                                    if(lU > 1)
                                        buffer(coord.asStdArray()) = val-1;
                                    else
                                        buffer(coord2.asStdArray()) = val-1;
                                }
                            }
                        }
                    }
                }
                
            });
            --val;
        }

        tools::parallelForEachCoordinate(threadpool, shape,
        [&](const int tid, const CoordType & coord){
            processedGroundTruth(coord.asStdArray()) = std::max(int64_t(0), buffer(coord.asStdArray()));
        });

        std::cout<<"region growing\n";
        // do region growing

        tools::VerboseTimer t(true, "region growing");
        t.startAndPrint();
        nifty::region_growing::seededRegionGrowing(integralGrowMap, processedGroundTruth,numberOfQueues);
        //nifty::region_growing::seededRegionGrowing2<DIM>(integralGrowMap, processedGroundTruth,numberOfQueues);
        t.stopAndPrint();
        std::cout<<"done\n";


        if(verbose)
            std::cout<<"finish maps\n";
        // fill out with initial segmentation value + 1
        // and prepare integral grow map
        tools::parallelForEachCoordinate(threadpool, shape,
        [&](const int tid, const CoordType & coord){
            --processedGroundTruth(coord.asStdArray());
        });
    }




}
}

#endif // NIFTY_GROUND_TRUTH_POST_PROCESS_CARVING_GROUND_TRUTH_HXX
