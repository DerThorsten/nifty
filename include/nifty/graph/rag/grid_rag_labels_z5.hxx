#pragma once

#include <algorithm>
#include <cstddef>

#include "xtensor/xarray.hpp"
#include "z5/dataset.hxx"
#include "z5/multiarray/xt_access.hxx"

#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/tools/block_access.hxx"

namespace nifty{
namespace graph{

template<std::size_t DIM, class LABEL_TYPE>
class Z5Labels{
public:
    typedef LABEL_TYPE LabelType;
    typedef tools::BlockStorage< LabelType> BlockStorageType;
    typedef const z5::Dataset Z5ArrayType;

    Z5Labels(const Z5ArrayType & labels, const uint64_t numberOfLabels)
    :   labels_(labels),
        shape_(),
        numberOfLabels_(numberOfLabels) {
        for(std::size_t i=0; i<DIM; ++i)
            shape_[i] = labels_.shape(i);
    }


    // part of the API
    uint64_t numberOfLabels() const {
        return numberOfLabels_;
    }
    const array::StaticArray<int64_t, DIM> & shape() const{
        return  shape_;
    }

    template<class ROI_BEGIN_COORD, class ROI_END_COORD>
    void readSubarray(
        const ROI_BEGIN_COORD & roiBeginCoord,
        const ROI_END_COORD & roiEndCoord,
        xt::xarray<LABEL_TYPE> & outArray) const{
        for(auto d = 0 ; d<DIM; ++d){
            NIFTY_CHECK_OP(roiEndCoord[d] - roiBeginCoord[d], ==,outArray.shape(d),"wrong shape");
            NIFTY_CHECK_OP(roiEndCoord[d], <=, labels_.shape()[d], "hubs");
        }
        z5::multiarray::readSubarray<LABEL_TYPE>(labels_, outArray, roiBeginCoord.begin());
    }

    Z5ArrayType & z5Array() const {
        return labels_;
    }

private:
    array::StaticArray<int64_t, DIM> shape_;
    Z5ArrayType & labels_;
    uint64_t numberOfLabels_;
};


} // namespace graph


namespace tools{

    template<class LABEL_TYPE, std::size_t DIM, class COORD>
    inline void readSubarray(
        const graph::Hdf5Labels<DIM, LABEL_TYPE> & labels,
        const COORD & beginCoord,
        const COORD & endCoord,
        xt::xarray<LABEL_TYPE> & subarray
    ) {
        z5::multiarray::readSubarray<LABEL_TYPE>(labels_, subarray, beginCoord.begin());
    }

}



} // namespace nifty
