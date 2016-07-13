#pragma once
#ifndef NIFTY_GRAPH_COMPONENTS_HXX
#define NIFTY_GRAPH_COMPONENTS_HXX

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty{
namespace graph{


template<class GRAPH>
class ComponentsUfd {

public:
    typedef GRAPH Graph;

    ComponentsUfd(const Graph & graph)
    :   graph_(graph),
        ufd_(graph.nodeIdUpperBound()+1),
        offset_(ufd_.numberOfElements() - graph_.numberOfNodes()),
        needsReset_(false){
        
    }

    uint64_t build(){
        return build(DefaultSubgraphMask<Graph>());
    }

    template<class SUBGRAPH_MASK>
    uint64_t build(const SUBGRAPH_MASK & mask){
        if(needsReset_)
            this->reset();
        for(auto edge : graph_.edges()){
            if(mask.useEdge(edge)){
                const auto u = graph_.u(edge);
                const auto v = graph_.v(edge);
                if(mask.useNode(u) && mask.useNode(v)){
                    ufd_.merge(u,v);
                }
            }
        }
        needsReset_ = true;
        return ufd_.numberOfSets() - offset_;
    }

    void reset(){
        ufd_.reset();
    }

    bool areConnected(const int64_t u, const int64_t v) const{
        return ufd_.find(u) == ufd_.find(v);
    }

    bool areConnected(const int64_t u, const int64_t v) {
        return ufd_.find(u) == ufd_.find(v);
    }

    uint64_t componentLabel(const uint64_t u) const{
        return ufd_.find(u);
    }

    uint64_t componentLabel(const uint64_t u){
        return ufd_.find(u);
    }

private:
    const Graph & graph_;
    nifty::ufd::Ufd< > ufd_;
    uint64_t offset_;
    bool needsReset_;
};


} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_COMPONENTS_HXX*/
