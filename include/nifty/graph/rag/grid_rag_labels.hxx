#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{

template<unsigned int DIM, class LABEL_TYPE>
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

    const std::array<int64_t, DIM> & shape()const{
        return  shape_;
    }

private:
    nifty::marray::View<LABEL_TYPE> labels_;
    std::array<int64_t, DIM> shape_;
};

}
}


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_LABELS_HXX */
