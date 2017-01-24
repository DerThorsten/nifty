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

        const VectorType & blocksPerAxisStrides() const {
            return blocksPerAxisStrides_;  
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
            return  BlockWithHaloType(BlockType(outerBegin, outerEnd), innerBlock);
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