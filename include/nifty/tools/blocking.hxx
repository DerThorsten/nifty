#pragma once

#include "nifty/"
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

        const Block(
            const VectorType & begin = VectorType(0),
            const VectorType & end = VectorType(0)
        )
        :   begin_(begin),
            end_(end){
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
        typedef typename BlockType::ValueType VectorType;

        const BlockWithHalo(
            const BlockType & outerBlock = BlockType(),
            const BlockType & innerBlock = BlockType()
        )
        :   outerBlock_(outerBlock),
            innerBlock_(innerBlock){
        }
    private:
        BlockType outerBlock_;
        BlockType innerBlock_;
    };




    template<std::size_t DIM, class T = int64_t>
    class Blocking{
    public:
        typedef BlockWithHalo<DIM, T> BlockWithHaloType;
        typedef typename BlockType::BlockType BlockType;
        typedef typename BlockType::ValueType ValueType;
        typedef typename BlockType::ValueType value_type;
        typedef typename BlockType::ValueType VectorType;

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
                const auto dimSize = roiEnd_ - (roiBegin_[d] - blockShift_[d]);
                const auto bs = blockShape_[d];
                const auto bpa =  dimSize / bs + int(dimSize % bs);
                blocksPerAxis_[d] = bpa;
                numberOfBlocks_ *= bpa;
            }
            
            blocksPerAxisStrides_[DIM - 1] = 1;
            for(size_t d = DIM-2; d>=0; --d){
                blocksPerAxisStrides_[d] = blocksPerAxisStrides_[d+1] * shape[d+1]
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
        
        const size_t numberOfBlocks()const{
            return numberOfBlocks_;
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
    }




}
}