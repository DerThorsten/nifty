#pragma once

#include "nifty/array/arithmetic_array.hxx"


namespace nifty{
namespace tools{


    template<std::size_t DIM, class T = int64_t>
    class Block{
    public:
        typedef T ValueType;
        /// stl compatible value type
        typedef ValueType value_type;
        typedef nifty::array::StaticArray<ValueType, DIM> VectorType;

        Block(
            const VectorType & begin = VectorType(0),
            const VectorType & end = VectorType(0)
        )
        :   begin_(begin),
            end_(end){
        }

        const VectorType & begin() const {
            return begin_;
        }

        const VectorType & end() const {
            return end_;
        }

        VectorType shape() const {
            return end_ - begin_;
        }

    private:
        VectorType begin_;
        VectorType end_;
    };



    template<std::size_t DIM, class T = int64_t>
    class BlockWithHalo{
    public:
        typedef Block<DIM, T> BlockType;
        typedef typename BlockType::ValueType ValueType;
        typedef typename BlockType::ValueType value_type;
        typedef typename BlockType::VectorType VectorType;


        BlockWithHalo(
            const BlockType & outerBlock = BlockType(),
            const BlockType & innerBlock = BlockType()
        )
        :   outerBlock_(outerBlock),
            innerBlock_(innerBlock),
            innerBlockLocal_(){

            const auto lBegin = innerBlock.begin()  - outerBlock.begin();
            const auto lEnd = lBegin  + innerBlock_.shape();
            innerBlockLocal_ = BlockType(lBegin, lEnd);
        }

        const BlockType & outerBlock() const {
            return outerBlock_;
        }

        const BlockType & innerBlock() const {
            return innerBlock_;
        }

        const BlockType & innerBlockLocal() const {
            return innerBlockLocal_;
        }


    private:
        BlockType outerBlock_;
        BlockType innerBlock_;
        BlockType innerBlockLocal_;
    };




    template<std::size_t DIM, class T = int64_t>
    class Blocking{
    public:
        typedef BlockWithHalo<DIM, T> BlockWithHaloType;
        typedef typename BlockWithHaloType::BlockType BlockType;

        typedef typename BlockWithHaloType::ValueType ValueType;
        typedef typename BlockWithHaloType::ValueType value_type;
        typedef typename BlockWithHaloType::VectorType VectorType;

        Blocking(
            const VectorType & roiBegin ,
            const VectorType & roiEnd,
            const VectorType & blockShape,
            const VectorType & blockShift = VectorType(0)
        )
        :   roiBegin_(roiBegin),
            roiEnd_(roiEnd),
            blockShape_(blockShape),
            blockShift_(blockShift),
            blocksPerAxis_(),
            blocksPerAxisStrides_(),
            numberOfBlocks_(1){

            for(size_t d=0; d<DIM; ++d){
                const auto dimSize = roiEnd_[d] - (roiBegin_[d] - blockShift_[d]);
                const auto bs = blockShape_[d];
                const auto bpa =  dimSize / bs + int((dimSize % bs) != 0);
                blocksPerAxis_[d] = bpa;
                numberOfBlocks_ *= bpa;
            }

            blocksPerAxisStrides_[DIM - 1] = 1;
            for(int64_t d = DIM-2; d>=0; --d){
                blocksPerAxisStrides_[d] = blocksPerAxisStrides_[d+1] * blocksPerAxis_[d+1];
            }
        }

        int64_t getNeighborId(const uint64_t blockId, const unsigned axis, const bool lower) const {

            const auto blockPosAtAxis = getBlockAxisPosition(blockId, axis);

            // we don't have lower neighbors for the lowest block in axis
            // and we don't have upper neighbor for the highest block in axis
            if(lower && blockPosAtAxis == 0) {
                return -1;
            } else if(!lower && blockPosAtAxis == blocksPerAxis_[axis] - 1) {
                return -1;
            }

            const auto stride = blocksPerAxisStrides_[axis];
            int64_t neighborId = blockId + (lower ? -stride : stride);
            //return (neighborId < numberOfBlocks_) ? (neighborId >= 0 ? neighborId : -1) : -1;
            return neighborId;
        }

        const VectorType & roiBegin() const {
            return roiBegin_;
        }

        const VectorType & roiEnd() const {
            return roiEnd_;
        }

        const VectorType & blockShape() const {
            return blockShape_;
        }

        const VectorType & blockShift() const {
            return blockShift_;
        }

        const VectorType & blocksPerAxis() const {
            return blocksPerAxis_;
        }

        const size_t numberOfBlocks()const{
            return numberOfBlocks_;
        }


        BlockType getBlock(const uint64_t blockIndex)const{

            // convert blockindex to coordinate
            uint64_t index = blockIndex;
            VectorType beginCoord, endCoord;
            for(auto d=0; d<DIM; ++d){

                const int64_t blockCoordAtD = index / blocksPerAxisStrides_[d];
                index -= blockCoordAtD*blocksPerAxisStrides_[d];

                const int64_t beginCoordAtD = (roiBegin_[d] - blockShift_[d]) + blockCoordAtD*blockShape_[d];
                endCoord[d]   =  std::min(beginCoordAtD + blockShape_[d], roiEnd_[d]);
                beginCoord[d] =  std::max(beginCoordAtD, roiBegin_[d]);
            }

            return BlockType(beginCoord, endCoord);
        }


        BlockWithHaloType getBlockWithHalo(
            const uint64_t blockIndex,
            const VectorType & haloBegin,
            const VectorType & haloEnd
        )const{
            const BlockType innerBlock = getBlock(blockIndex);

            VectorType outerBegin,outerEnd;

            for(auto d=0; d<DIM; ++d){
                outerBegin[d] = std::max(innerBlock.begin()[d] - haloBegin[d], roiBegin_[d]);
                outerEnd[d]   = std::min(innerBlock.end()[d]   + haloEnd[d], roiEnd_[d]);
            }
            return BlockWithHaloType(BlockType(outerBegin, outerEnd), innerBlock);
        }


        // get all block ids that are enclosed in the roi
        void getBlockIdsInBoundingBox(
                const VectorType & roiBegin,
                const VectorType & roiEnd,
                const VectorType & blockHalo,
                std::vector<uint64_t> & idsOut) const {

            // TODO assert that the roi is in global roi

            idsOut.clear();

            for(size_t blockId = 0; blockId < numberOfBlocks(); ++blockId) {

                // get coordinates of the current bock
                const auto & block = getBlockWithHalo(blockId, blockHalo).outerBlock();
                const auto & begin = block.begin();
                const auto & end   = block.end();

                // check for each dimension whether the current block has overlap with the roi
                std::vector<bool> enclosedInDim(DIM, false);
                for( auto d = 0; d < DIM; ++d) {
                    if(begin[d] >= roiBegin[d] && end[d] <= roiEnd[d]) {
                        enclosedInDim[d] = true;
                    }
                }

                // if all dimentsions have overlap, push back the block id
                if(std::all_of(enclosedInDim.begin(), enclosedInDim.end(), [](bool i){return i;})) {
                    idsOut.push_back(blockId);
                }
            }
        }


        // get all block ids that have overlap with the roi
        void getBlockIdsOverlappingBoundingBox(
                const VectorType & roiBegin,
                const VectorType & roiEnd,
                const VectorType & blockHalo,
                std::vector<uint64_t> & idsOut) const {

            // TODO assert that the roi is in global roi

            idsOut.clear();

            // lambda to check whether two values are in range
            auto valueInRange = [](T value, T min, T max) {
                return (value >= min) && (value <= max);
            };

            for(size_t blockId = 0; blockId < numberOfBlocks(); ++blockId) {

                // get coordinates of the current bock
                const auto & block = getBlockWithHalo(blockId, blockHalo).outerBlock();
                const auto & begin = block.begin();
                const auto & end   = block.end();

                // check for each dimension whether the current block has overlap with the roi
                std::vector<bool> overlapInDim(DIM, false);
                for( auto d = 0; d < DIM; ++d) {
                    if( valueInRange(roiBegin[d], begin[d], end[d]) || valueInRange(begin[d], roiBegin[d], roiEnd[d]) ) {
                        overlapInDim[d] = true;
                    }
                }

                // if all dimentsions have overlap, push back the block id
                if(std::all_of(overlapInDim.begin(), overlapInDim.end(), [](bool i){return i;})) {
                    idsOut.push_back(blockId);
                }
            }
        }



        // return the overlaps (in local block coordinates for two specified blocks)
        bool getLocalOverlaps(
                const uint64_t blockAId,
                const uint64_t blockBId,
                const VectorType & blockHalo,
                VectorType & overlapBeginA,
                VectorType & overlapEndA,
                VectorType & overlapBeginB,
                VectorType & overlapEndB
        ) const {

            // lambda to check whether two values are in range
            auto valueInRange = [](T value, T min, T max) {
                return (value >= min) && (value <= max);
            };

            // TODO use std::bitset instead ?
            // determine whether the query block starts inside block
            auto isLeft = [&](const BlockType & queryBlock, const BlockType & block) {
                std::vector<bool> left(DIM, false);
                const auto & queryBegin = queryBlock.begin();
                const auto & begin = block.begin();
                const auto & end   = block.end();
                for(int d = 0; d < DIM; ++d) {
                    left[d] = valueInRange(queryBegin[d], begin[d], end[d]);
                }
                return left;
            };

            const auto blockA = getBlockWithHalo(blockAId, blockHalo).outerBlock();
            const auto blockB = getBlockWithHalo(blockBId, blockHalo).outerBlock();

            auto aIsLeft = isLeft(blockA, blockB);
            auto bIsLeft = isLeft(blockB, blockA);

            std::vector<bool> overlaps(DIM);
            for(int d = 0; d < DIM; ++d) {
                overlaps[d] = aIsLeft[d] || bIsLeft[d];
            }

            const auto & beginA = blockA.begin();
            const auto & beginB = blockB.begin();
            const auto & endA = blockA.end();
            const auto & endB = blockB.end();

            VectorType globalOverlapBegin, globalOverlapEnd;
            // check if the blocks are overlapping
            if( std::all_of( overlaps.begin(), overlaps.end(), [](bool i){return i;}) ) {

                // set the appropriate begin and end for each dimension
                for(int d = 0; d < DIM; ++d) {
                    // a is left in this dimension -> we set the beginning to begin(A) end the end to end(B)
                    if(aIsLeft[d]) {
                        globalOverlapBegin[d] = beginA[d];
                        globalOverlapEnd[d] = endB[d];
                    }
                    else { // b is left in this dimension, or a and b are equal -> we set the beginning to begin(B) and the end to end(A)
                        globalOverlapBegin[d] = beginB[d];
                        globalOverlapEnd[d] = endA[d];
                    }
                }

            }
            else { // otherwise return that no overlap was found
                return false;
            }

            for(int d = 0; d < DIM; ++d) {
                overlapBeginA[d] = globalOverlapBegin[d] - beginA[d];
                overlapEndA[d] = globalOverlapEnd[d] - beginA[d];

                overlapBeginB[d] = globalOverlapBegin[d] - beginB[d];
                overlapEndB[d] = globalOverlapEnd[d] - beginB[d];
            }

            return true;
        }


        // get all block ids in slice z
        void getBlockIdsInSlice(
                const T z,
                const VectorType & blockHalo,
                std::vector<uint64_t> & idsOut) const {

            // TODO assert that the roi is in global roi
            //
            idsOut.clear();

            for(size_t blockId = 0; blockId < numberOfBlocks(); ++blockId) {
                const auto & block = getBlockWithHalo(blockId, blockHalo).outerBlock();
                const auto & begin = block.begin();
                const auto & end   = block.end();
                // check if this slice is in z
                auto z_start = begin[0];
                auto z_end   = end[0];
                if(z >= z_start && z < z_end )
                    idsOut.push_back(blockId);
            }
        }


        BlockWithHaloType getBlockWithHalo(
            const uint64_t blockIndex,
            const VectorType & halo
        )const{
            return this->getBlockWithHalo(blockIndex, halo, halo);
        }


        BlockWithHaloType addHalo(
            const BlockType innerBlock,
            const VectorType & haloBegin,
            const VectorType & haloEnd
        )const{

            VectorType outerBegin,outerEnd;

            for(auto d=0; d<DIM; ++d){
                outerBegin[d] = std::max(innerBlock.begin()[d] - haloBegin[d], roiBegin_[d]);
                outerEnd[d]   = std::min(innerBlock.end()[d]   + haloEnd[d], roiEnd_[d]);
            }
            return  BlockWithHaloType(BlockType(outerBegin, outerEnd), innerBlock);
        }



        BlockWithHaloType addHalo(
            const BlockType innerBlock,
            const VectorType & halo
        )const{
            return this->addHalo(innerBlock, halo, halo);
        }


    private:

        uint64_t getBlockAxisPosition(const uint64_t blockId, const unsigned axis) const {
            // get the position of the block in this axis
            uint64_t index = blockId;
            int64_t blockPosAtAxis;
            for(auto d = 0; d < DIM; ++d) {
                blockPosAtAxis = index / blocksPerAxisStrides_[d];
                index -= blockPosAtAxis * blocksPerAxisStrides_[d];
                if(d == axis){
                    break;
                }
            }
            return blockPosAtAxis;
        }

        // given from user
        VectorType roiBegin_;
        VectorType roiEnd_;
        VectorType blockShape_;
        VectorType blockShift_;

        VectorType blocksPerAxis_;
        VectorType blocksPerAxisStrides_;
        size_t numberOfBlocks_;
    };




}
}
