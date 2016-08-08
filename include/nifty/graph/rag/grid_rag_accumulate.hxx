#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

#include "vigra/accumulator.hxx"

namespace nifty{
namespace graph{


    template<class RAG, class DATA>
    void accumulateMean(
        const RAG & rag,
        const DATA & data
    ){
        using namespace vigra::acc;
        StandAloneAccumulatorChain<>
    }
    

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX */
