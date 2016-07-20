#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"
//
#include "vigra/multi_array_chunked_hdf5.hxx"


namespace nifty{
namespace graph{

template<size_t DIM, class LABEL_TYPE>
class ExplicitLabels{
public:

    typedef nifty::marray::View<LABEL_TYPE> ViewType;

    ExplicitLabels(const nifty::marray::View<LABEL_TYPE, false> & labels = nifty::marray::View<LABEL_TYPE, false>())
    :   labels_(labels),
        shape_()
    {
        for(size_t i=0; i<DIM; ++i)
            shape_[i] = labels_.shape(i);
    }


    // part of the API
    uint64_t numberOfLabels() const {
        auto  startPtr = &labels_(0);
        auto  lastElement = &labels_(labels_.size()-1);
        auto d = lastElement - startPtr + 1;

        if(d == labels_.size()){
            return *std::max_element(startPtr, startPtr+labels_.size())+1;
        }
        else if(labels_.isSimple()){
            
            NIFTY_CHECK_OP(d,==,labels_.size(),"");
            return *std::max_element(startPtr, startPtr+labels_.size())+1;
        }
        else {
            LABEL_TYPE nLabels = 0;
            for(size_t i=0; i<labels_.size(); i++)
                nLabels = std::max(labels_(i), nLabels);
            return nLabels+1;
        }
    }

    // not part of the general API
    const ViewType & labels() const{
        return labels_;
    }

    const std::array<int64_t, DIM> & shape() const{
        return  shape_;
    }

private:
    ViewType labels_;
    std::array<int64_t, DIM> shape_;
};



template<size_t DIM, class LABEL_TYPE>
class ChunkedLabels{
public:

    typedef vigra::ChunkedArrayHDF5<DIM, LABEL_TYPE> ViewType;

    // enable setting the chunk size by overloading the constuctor
    ChunkedLabels(const std::string & label_file, const std::string & label_key )
    : labels_(vigra::HDF5File(label_file, vigra::HDF5File::ReadOnly), label_key ),
      shape_(), label_file_(label_file), label_key_(label_key)
    {
        for(size_t i=0; i<DIM; ++i)
            shape_[i] = labels_.shape(i);
    }

    // need the copy constructor!
    ChunkedLabels(const ChunkedLabels & src)
    : labels_(vigra::HDF5File(src.label_file_, vigra::HDF5File::ReadOnly), src.label_key_ ),
      shape_(src.shape_), label_file_(src.label_file_), label_key_(src.label_key_)
    {}

    
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
    ViewType labels_;
    std::array<int64_t, DIM> shape_;
    std::string label_file_;
    std::string label_key_;

};

} // namespace graph
} // namespace nifty




#endif /* NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HXX */
