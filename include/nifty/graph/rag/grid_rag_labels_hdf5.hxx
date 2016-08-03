#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HDF5_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HDF5_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/hdf5/hdf5_array.hxx"

namespace nifty{
namespace graph{

template<size_t DIM, class LABEL_TYPE>
class Hdf5Labels{
public:

    typedef marray::Marray<LABEL_TYPE> SubarrayViewType;

    Hdf5Labels()
    {

    }


    // part of the API
    uint64_t numberOfLabels() const {

    }
    const array::StaticArray<int64_t, DIM> & shape()const{
        return  shape_;
    }

    template<
        class ROI_BEGIN_COORD,
        class ROI_END_COORD
    >
    void readSubarray(
        const ROI_BEGIN_COORD & roiBeginCoord,
        const ROI_END_COORD & roiEndCoord,
        marray::View<LABEL_TYPE> & outArray
    )const{
        labels_.readSubarray(roiBeginCoord.begin(), outArray);
    }

    

private:
    array::StaticArray<int64_t, DIM> shape_;
    hdf5::Hdf5Array<LABEL_TYPE> labels_;
};


} // namespace graph
} // namespace nifty




#endif /* NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HDF5_HXX */
