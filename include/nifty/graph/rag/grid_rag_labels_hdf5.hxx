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

    typedef const hdf5::Hdf5Array<LABEL_TYPE> Hdf5ArrayType;

    Hdf5Labels(const Hdf5ArrayType & labels, const uint64_t numberOfLabels)
    :   labels_(labels),
        shape_(),
        numberOfLabels_(numberOfLabels)
    {
        for(size_t i=0; i<DIM; ++i)
            shape_[i] = labels_.shape(i);
    }


    // part of the API
    uint64_t numberOfLabels() const {
        return numberOfLabels_;
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
        for(auto d = 0 ; d<DIM; ++d){
            NIFTY_CHECK_OP(roiEndCoord[d] - roiBeginCoord[d],==,outArray.shape(d),"wrong shape");
        }
        labels_.readSubarray(roiBeginCoord.begin(), outArray);
    }


    const hdf5::Hdf5Array<LABEL_TYPE> & hdf5Array()const{
        return labels_;
    }

private:
    array::StaticArray<int64_t, DIM> shape_;
    const hdf5::Hdf5Array<LABEL_TYPE> & labels_;
    int64_t numberOfLabels_;
};


} // namespace graph
} // namespace nifty




#endif /* NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HDF5_HXX */
