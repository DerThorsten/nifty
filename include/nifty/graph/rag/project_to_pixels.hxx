#pragma once
#ifndef NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX
#define NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{



template<class LABELS_TYPE, class LABELS, class NODE_MAP>
void gridRagAccumulateLabels(
    const ExplicitLabelsGridRag<2, LABELS_TYPE> & graph,
    nifty::marray::View<LABELS> data,
    NODE_MAP &  nodeMap
){
    const auto labelsProxy = graph.labelsProxy();
    const auto labels = labelsProxy.labels(); 

    std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());
 
    for(size_t x=0; x<labels.shape(0); ++x)
    for(size_t y=0; y<labels.shape(1); ++y){
        const auto node = labels(x, y);            
        const auto l  = data(x,y);
        overlaps[node][l] += 1;
    }
    for(const auto node : graph.nodes()){
        const auto & ol = overlaps[node];
        // find max ol 
        uint64_t maxOl = 0 ;
        uint64_t maxOlLabel = 0;
        for(auto kv : ol){
            if(kv.second > maxOl){
                maxOl = kv.second;
                maxOlLabel = kv.first;
            }
        }
        nodeMap[node] = maxOlLabel;
    }
}


}
}


#endif /* NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX */
