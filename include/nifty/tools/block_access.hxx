#ifndef NIFTY_TOOLS_BLOCK_ACCESS_HXX
#define NIFTY_TOOLS_BLOCK_ACCESS_HXX

#include "nifty/marray/marray.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace tools{


template<size_t DIM, class T>
class BlockStorage{
public:
    typedef marray::Marray<T> ArrayType;
    typedef marray::View<T> ViewType;
    typedef array::StaticArray<int64_t, DIM> ShapeType;

    template<class SHAPE>
    BlockStorage(
        const SHAPE & maxShape,  
        const std::size_t numberOfBlocks
    )
    :   arrayVec_(numberOfBlocks, ArrayType(maxShape.begin(), maxShape.end())){
        std::fill(zeroCoord_.begin(), zeroCoord_.end(), 0);
    }

    template<class SHAPE>
    BlockStorage(
        nifty::parallel::ThreadPool & threadpool,
        const SHAPE & maxShape,  
        const std::size_t numberOfBlocks
    )
    :   arrayVec_(numberOfBlocks, ArrayType(maxShape.begin(), maxShape.end())){
        std::fill(zeroCoord_.begin(), zeroCoord_.end(), 0);
    }

    template<class SHAPE>
    ViewType resizeIfNecessary(const SHAPE & shape, const std::size_t blockIndex) {
        return arrayVec_[blockIndex].view(zeroCoord_.begin(), shape.begin());
    }

private:
    ShapeType zeroCoord_;
    std::vector<ArrayType> arrayVec_;
};

template<size_t DIM, class T>
class BlockView{
public:
    typedef marray::View<T> ViewType;
    typedef array::StaticArray<int64_t, DIM> ShapeType;

    template<class SHAPE>
    BlockView(
        const SHAPE & maxShape,  
        const std::size_t numberOfBlocks
    ){

    }

    template<class SHAPE>
    BlockView(
        nifty::parallel::ThreadPool & threadpool,
        const SHAPE & maxShape,  
        const std::size_t numberOfBlocks
    ){

    }


    template<class SHAPE>
    ViewType resizeIfNecessary(const SHAPE & shape, const std::size_t blockIndex) {
        return ViewType();
    }

private:
    //std::vector<ViewType> viewVec_;
};


} // end namespace nifty::tools
} // end namespace nifty

#endif /*NIFTY_TOOLS_BLOCK_ACCESS_HXX*/
