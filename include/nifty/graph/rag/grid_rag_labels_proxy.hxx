#pragma once

#include <algorithm>
#include <cstddef>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/tools/block_access.hxx"

#include "nifty/xtensor/xtensor.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

// FIXME this causes multiple definition linker errors
// why ?!
#ifdef WITH_Z5
#include "nifty/z5/z5.hxx"
#endif


namespace nifty{
namespace graph{

template<std::size_t DIM, class LABEL_ARRAY>
class LabelsProxy {
public:
    typedef LABEL_ARRAY LabelArrayType;
    // this should also work for xtensor
    typedef typename LabelArrayType::value_type LabelType;
    typedef tools::BlockStorage<LabelType> BlockStorageType;

    LabelsProxy(
        const LabelArrayType & labels,
        const std::size_t numberOfLabels
    )
    :   labels_(labels),
        numberOfLabels_(numberOfLabels),
        shape_()
    {
        auto & tmpShape = labels_.shape();
        for(std::size_t i=0; i<DIM; ++i) {
            shape_[i] = tmpShape[i];
        }
    }

    std::size_t numberOfLabels() const {
        return numberOfLabels_;
    }

    template<class ROI_BEGIN_COORD, class ROI_END_COORD, class ARRAY>
    void readSubarray(const ROI_BEGIN_COORD & roiBeginCoord,
                      const ROI_END_COORD & roiEndCoord,
                      xt::xexpression<ARRAY> & outArray) const {
        tools::readSubarray(labels_, roiBeginCoord, roiEndCoord, outArray);
    }

    const LabelArrayType & labels() const{
        return labels_;
    }

    const array::StaticArray<int64_t, DIM> & shape() const{
        return  shape_;
    }

private:
    const LabelArrayType & labels_;
    std::size_t numberOfLabels_;
    array::StaticArray<int64_t, DIM> shape_;
};


} // namespace nifty::graph


namespace tools{

    template<class LABEL_ARRAY, std::size_t DIM, class COORD, class ARRAY>
    inline void readSubarray(const graph::LabelsProxy<DIM, LABEL_ARRAY> & labels,
                             const COORD & beginCoord,
                             const COORD & endCoord,
                             xt::xexpression<ARRAY> & subarray) {
        labels.readSubarray(beginCoord, endCoord, subarray);
    }

} // namespace nifty::tools
} // namespace nifty
