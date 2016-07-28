#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_LABELS_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_LABELS_CHUNKED_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"
//
#include "vigra/multi_array_chunked_hdf5.hxx"

namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class ChunkedLabels{
public:

    // TODO try to use ChunkedArray here instead, s. t. we can use all the classes derived from it
    typedef vigra::ChunkedArrayHDF5<DIM, LABEL_TYPE> ViewType;

    // enable setting the chunk size by overloading the constuctor
    ChunkedLabels(const std::string & label_file, const std::string & label_key )
    : labels_(vigra::HDF5File(label_file, vigra::HDF5File::ReadOnly), label_key ),
      shape_(), label_file_(label_file), label_key_(label_key)
    {
        for(size_t i=0; i<DIM; ++i)
            shape_[i] = labels_.shape(i);
    }

    
    // part of the API
    // TODO iterate over the chunks in parellel !
    uint64_t numberOfLabels() const {
        return *std::max_element(labels_.cbegin(), labels_.cend())+1;
        //for(auto it = _labels.chunk_begin(); it != _labels.chunk_end; ++it ) P
        //  
        //}
    }

    // not part of the general API
    const ViewType & labels() const{
        return labels_;
    }

    const std::array<int64_t, DIM> & shape() const{
        return  shape_;
    }


private:
    std::array<int64_t, DIM> shape_;
    std::string label_file_;
    std::string label_key_;
    ViewType labels_;

};



} // namespace graph
} // namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_LABELS_CHUNKED_HXX */
