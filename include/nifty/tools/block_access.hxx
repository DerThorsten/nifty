#pragma once

#include "xtensor/xarray.hpp"
#include "nifty/xtensor/xtensor.hxx"

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace tools{


template<class T>
class BlockStorage{
public:
    typedef xt::xarray<T> ArrayType;
    template<class SHAPE>
    BlockStorage(
        const SHAPE & maxShape,
        const std::size_t numberOfBlocks
    )
    :   arrayVec_(numberOfBlocks, ArrayType(maxShape)){
        std::fill(zeroCoord_.begin(), zeroCoord_.end(), 0);
    }

    template<class SHAPE>
    BlockStorage(
        nifty::parallel::ThreadPool & threadpool,
        const SHAPE & maxShape,
        const std::size_t numberOfBlocks
    )
    :   arrayVec_(numberOfBlocks),
        zeroCoord_(maxShape.size(),0)
    {
        std::vector<std::size_t> arrayShape(maxShape.begin(), maxShape.end());
        nifty::parallel::parallel_foreach(threadpool, numberOfBlocks, [&](const int tid, const int i){
            arrayVec_[i] = ArrayType(arrayShape);
        });
    }

    template<class SHAPE>
    inline auto getView(const SHAPE & shape, const std::size_t blockIndex) {
        auto & array = arrayVec_[blockIndex];
        xt::xstrided_slice_vector slice;
        xtensor::sliceFromRoi(slice, zeroCoord_, shape);
        return xt::strided_view(array, slice);
    }

    inline auto & getView(const std::size_t blockIndex) {
        return arrayVec_[blockIndex];
    }

private:
    std::vector<uint64_t> zeroCoord_;
    std::vector<ArrayType> arrayVec_;
};


} // end namespace nifty::tools
} // end namespace nifty

