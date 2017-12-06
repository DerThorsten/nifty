#pragma once


#include <algorithm>
#include <cstddef>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/tools/block_access.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{

template<std::size_t DIM, class LABEL_ARRAY>
class LabelsProxy {
public:
    typedef LABEL_ARRAY LabelArrayType;
    // TODO get the data type from the label array
    typedef typename LabelArrayType::DataType LabelType
    typedef tools::BlockView< LabelType> BlockStorageType;

    // TODO switch to xtensor
    ExplicitLabels(
        const LabelArrayType & labels,
        const std::size_t numberOfLabels
    )
    :   labels_(labels),
        numberOfLabels_(numberOfLabels),
        shape_()
    {
        // FIXME this probably won't work for xt shape
        for(std::size_t i=0; i<DIM; ++i)
            shape_[i] = labels_.shape(i);
    }

    std::size_t numberOfLabels() const {
        return numberOfLabels_;
    }

    // TODO switch to xtensor
    template<class ROI_BEGIN_COORD, class ROI_END_COORD>
    void readSubarray(const ROI_BEGIN_COORD & roiBeginCoord,
                      const ROI_END_COORD & roiEndCoord,
                      marray::View<LabelType> & outArray) const {
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

    // TODO switch to xtensor
    template<class LABEL_ARRAY, std::size_t DIM, class COORD>
    inline void readSubarray(const graph::LabelsProxy<DIM, LABEL_ARRAY> & labels,
                             const COORD & beginCoord,
                             const COORD & endCoord,
                             marray::View<typename LABEL_ARRAY::DataType> & subarray) {
        labels.readSubarray(beginCoord, endCoord, subarray);
    }

} // namespace nifty::tools




} // namespace nifty
