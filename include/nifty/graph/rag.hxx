#pragma once
#ifndef NIFTY_COMPUTE_RAG_HXX
#define NIFTY_COMPUTE_RAG_HXX

#include "nifty/graph/simple_graph.hxx"
#include "nifty/marray/marray.hxx"

namespace nifty{
namespace graph{


    struct ComputeRagSettings{
        int numberOfThreads = -1;
        bool stacked2DVoxels = false;
    };

    template<class E, class N, class T>
    void computeRag(
        UndirectedGraph<E,N> & graph,  
        nifty::marray::View<T> & labels,
        const ComputeRagSettings & settings = ComputeRagSettings()
    ){
        const auto nDim = labels.dimension();
        if(nDim == 2){

        }
        else if(nDim == 3){

        } 
        else{

        }
    }
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_COMPUTE_RAG_HXX
