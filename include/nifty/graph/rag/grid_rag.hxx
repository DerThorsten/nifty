#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_HXX


#include <random>
#include <functional>

#include "nifty/marray/marray.hxx"
#include "nifty/graph/simple_graph.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{





template<class LABEL_TYPE>
class ExplicitLabels{
public:
    ExplicitLabels(const nifty::marray::View<LABEL_TYPE, false> & labels = nifty::marray::View<LABEL_TYPE, false>())
    :   labels_(labels){

    }
private:
    nifty::marray::View<LABEL_TYPE> labels_;
};


template<unsigned int DIM, class LABELS_PROXY>
class GridRag : public UndirectedGraph<>{
public:
    typedef LABELS_PROXY LabelsProxy;
    GridRag(const LabelsProxy & labelsProxy)
    :   lablesProxy_(labelsProxy){

    }
private:
    LabelsProxy lablesProxy_;
};


template<unsigned int DIM, class LABEL_TYPE>
using ExplicitLabelsGridRag = GridRag<DIM, ExplicitLabels<LABEL_TYPE> > ; 



}
}


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_HXX */