#pragma once

#include "nifty/tools/blocking.hxx"
#include "nifty/tools/array_tools.hxx"
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace tools{

    // TODO we should exclude defected slices
    template<class T, class COORD>
    void nodesToBlocksStacked(const nifty::hdf5::Hdf5Array<T> & segmentation,
            const Blocking<3> & blocking,
            const COORD & halo,
            const std::vector<int64_t> & skipSlices, // slices we skip due to defects
            std::vector<std::vector<T>> & out,
            const int nThreads = -1) {

        typedef tools::BlockStorage<T> LabelsStorage;
        typedef nifty::array::StaticArray<int64_t,3> Coord;
        typedef nifty::array::StaticArray<int64_t,2> Coord2;
        // copy halo to internal coordinate type
        Coord internalHalo({halo[0],halo[1],halo[2]});
    
        nifty::parallel::ParallelOptions pOpts(nThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        
        size_t nBlocks = blocking.numberOfBlocks();
        out.resize(nBlocks);

        const auto & shape = segmentation.shape();
        Coord sliceShape({1L,int64_t(shape[1]),int64_t(shape[2])});

        // thread data
        LabelsStorage labelsStorage(threadpool, sliceShape, nThreads);
        std::vector< std::map<uint64_t,std::vector<T> >> threadData(nThreads);
        
        // loop over the slices in parallel and find uniques for the blocking
        nifty::parallel::parallel_foreach(threadpool, shape[0], [&](const int64_t tid, const int64_t sliceId){

            if(std::find(skipSlices.begin(), skipSlices.end(), sliceId) != skipSlices.end())
                return;
            
            auto & threadUniques = threadData[tid];

            // get subblocks in this slice
            Coord sliceBegin({sliceId,0L,0L});
            std::vector<uint64_t> subBlocks;
            blocking.getBlockIdsInSlice(sliceId, internalHalo, subBlocks);
            
            // read segmentation in this slice
            auto subseg = labelsStorage.getView(tid);
            segmentation.readSubarray(sliceBegin.begin(), subseg);
            auto subsegSqueezed = subseg.squeezedView();
            
            // find uniques in subblocks
            Coord2 blockBegin, blockShape;
            std::vector<T> blockUniques;
            for(auto blockId : subBlocks) {
                const auto block = blocking.getBlockWithHalo(blockId, internalHalo).outerBlock();
                const auto & blockBegin3d = block.begin();
                const auto & blockShape3d = block.shape();
                size_t expSize = 1;
                for(int d = 0; d < 2; ++d) {
                    blockBegin[d] = blockBegin3d[d+1];
                    blockShape[d] = blockShape3d[d+1];
                }

                auto blockView = subsegSqueezed.view(blockBegin.begin(), blockShape.begin());
                uniques(blockView, blockUniques);
                // check whether the block is already in the map, extend if it is, inser otherwise
                // insertion method from http://stackoverflow.com/questions/97050/stdmap-insert-or-stdmap-find
                auto blockIt = threadUniques.lower_bound(blockId);
                if(blockIt != threadUniques.end() && !(threadUniques.key_comp()(blockId, blockIt->first))) { // key already there
                    //std::cout << "Update, block " << blockId << std::endl;
                    auto & blockVec = blockIt->second;
                    blockVec.insert(blockVec.end(), blockUniques.begin(), blockUniques.end());
                }
                else { // key not there, insert with hint
                    //std::cout << "Insert, block " << blockId << std::endl;
                    threadUniques.insert(blockIt, std::make_pair(blockId, blockUniques) );
                }
            }
        });

        // merge the threadUniques into the out vector
        nifty::parallel::parallel_foreach(threadpool, nBlocks, [&](const int64_t tid, const int64_t blockId){
            auto & outVec = out[blockId];
            std::vector<T> tmp;
            for(int threadId = 0; threadId < nThreads; ++threadId) {
                const auto & threadUniques = threadData[threadId];
                auto blockIt = threadUniques.find(blockId);
                if(blockIt != threadUniques.end()){
                    const auto & blockVec = blockIt->second;
                    tmp.insert(tmp.end(), blockVec.begin(), blockVec.end());
                }
            }
            uniques(tmp, outVec);
        });
    }
}
}
