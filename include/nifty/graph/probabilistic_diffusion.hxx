#pragma once


#include <algorithm> // sort


#include "vigra/priority_queue.hxx"
#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty{
namespace graph{



    template<
        class GRAPH,
        class CUT_PROBS,
        class CUT_PROBS_OUT
    >
    void edgeWeightedWatershedsSegmentationImpl(
        const GRAPH         & g,
        const CUT_PROBS  & cutProbs,
        CUT_PROBS_OUT    & cutProbsOut,
    ){  
        typedef GRAPH GraphType;
        typedef typename EDGE_WEIGHTS::value_type WeightType;

       

      

        
        // put edges from nodes with seed on pq
        for(auto node : g.nodes()){
            if(labels[node]!=static_cast<LabelType>(0)){

                for(auto adj : g.adjacency(node)){
                    const auto edge = adj.edge();
                    const auto neigbour = adj.node();
                    
                }
            }
        }

    }









} // namespace nifty::graph
} // namespace nifty

