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
class ExplicitLabels;

template<unsigned int DIM, class LABELS_PROXY>
class GridRag;





template<class LABEL_TYPE>
void computeRag(
    GridRag<2,  ExplicitLabels<LABEL_TYPE> > & rag
){



    const auto labelsProxy = rag.labelsProxy();
    const auto numberOfLabels = labelsProxy.numberOfLabels();
    const auto labels = labelsProxy.labels(); 
    
    //
    std::cout<<"nLabels "<<numberOfLabels<<"\n";
    // assign the number of nodes to the graph
    rag.assign(numberOfLabels);

    for(size_t x=0; x<labels.shape(0); ++x)
    for(size_t y=0; y<labels.shape(1); ++y){

        const auto lu = labels(x, y);
        if(x+1<labels.shape(0)){
            const auto lv = labels(x+1, y);
            if(lu != lv){
                rag.insertEdge(lu,lv);
            }
        }
        if(y+1<labels.shape(1)){
            const auto lv = labels(x, y+1);
            if(lu != lv){
                rag.insertEdge(lu,lv);
            }
        }
    }
}





template<class LABEL_TYPE>
class ExplicitLabels{
public:

    typedef nifty::marray::View<LABEL_TYPE> ViewType;

    ExplicitLabels(const nifty::marray::View<LABEL_TYPE, false> & labels = nifty::marray::View<LABEL_TYPE, false>())
    :   labels_(labels){

    }


    // part of the API
    uint64_t numberOfLabels() const {
        return *std::max_element(labels_.begin(), labels_.end())+1;
    }

    // not part of the general API
    const ViewType & labels() const{
        return labels_;
    }

private:
    nifty::marray::View<LABEL_TYPE> labels_;
};



template<unsigned int DIM, class LABELS_PROXY>
class GridRag : public UndirectedGraph<>{
public:
    typedef LABELS_PROXY LabelsProxy;
    GridRag(const LabelsProxy & labelsProxy)
    :   labelsProxy_(labelsProxy)
    {
        computeRag(*this);
    }
    const LabelsProxy & labelsProxy() const {
        return labelsProxy_;
    }
private:
    LabelsProxy labelsProxy_;
};


template<unsigned int DIM, class LABEL_TYPE>
using ExplicitLabelsGridRag = GridRag<DIM, ExplicitLabels<LABEL_TYPE> > ; 









}
}


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_HXX */