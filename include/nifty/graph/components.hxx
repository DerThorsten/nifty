#pragma once


#include <unordered_map>

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/ufd/ufd.hxx"


namespace nifty{
namespace graph{


template<class GRAPH>
class ComponentsUfd {

public:
    typedef GRAPH GraphType;

    ComponentsUfd(const GraphType & graph)
    :   graph_(graph),
        ufd_(graph.nodeIdUpperBound()+1),
        offset_(ufd_.numberOfElements() - graph_.numberOfNodes()),
        needsReset_(false){
        
    }

    uint64_t build(){
        return build(DefaultSubgraphMask<GraphType>());
    }

    template<class NODE_LABELS>
    uint64_t buildFromLabels(const NODE_LABELS & nodeLabels){
        if(needsReset_)
            this->reset();
        for(auto edge : graph_.edges()){
            const auto u = graph_.u(edge);
            const auto v = graph_.v(edge);
            if(nodeLabels[u] == nodeLabels[v]){
                ufd_.merge(u,v);
            }
        }
        needsReset_ = true;
        return ufd_.numberOfSets() - offset_;
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

    uint64_t operator[](const uint64_t u) const {
        return this->componentLabel(u);
    }
    
    uint64_t maxLabel()const{
        uint64_t maxLabel = 0;
        graph_.forEachNode([&](const uint64_t node){
            maxLabel = std::max(this->componentLabel(node), maxLabel);
        });
        return maxLabel; 
    }

    template<class NODE_MAP>
    void denseRelabeling(
        NODE_MAP & nodeMap
    )const{

        std::unordered_map<uint64_t,uint64_t> map;
        ufd_.representativeLabeling(map);

        for(const auto node : graph_.nodes()){
            nodeMap[node] = map[this->componentLabel(node)] - offset_;
        }
    
    }


    template<class NODE_MAP,class COMP_SIZE>
    void denseRelabeling(
        NODE_MAP & nodeMap,
        COMP_SIZE & compSize
    )const{

        std::unordered_map<uint64_t,uint64_t> map;
        ufd_.representativeLabeling(map);

        for(const auto node : graph_.nodes()){
            const auto denseLabel = map[this->componentLabel(node)] - offset_;
            compSize[denseLabel] += 1;
            nodeMap[node] = map[this->componentLabel(node)] - offset_;

        }
    
    }

    const GraphType & graph()const{
        return graph_;
    }



private:
    const GraphType & graph_;
    nifty::ufd::Ufd< > ufd_;
    uint64_t offset_;
    bool needsReset_;
};








template<class GRAPH>
class ComponentsBfs {

public:
    typedef GRAPH GraphType;
    typedef typename GraphType:: template EdgeMap<uint64_t> LabelsMapType;
    typedef typename GraphType:: template EdgeMap<bool>    VisitedMapType;

    ComponentsBfs(const GraphType & graph)
    :   graph_(graph),
        labels_(graph),
        visited_(graph, false),
        needsReset_(false),
        numberOfLabels_(0)
    {
        
    }

    uint64_t build(){
        return build(DefaultSubgraphMask<GraphType>());
    }

    template<class SUBGRAPH_MASK>
    uint64_t build(const SUBGRAPH_MASK & mask){
        if(needsReset_)
            this->reset();
        
        numberOfLabels_ = 0;



        

        graph_.forEachNode([&](const uint64_t node){
            if(mask.useNode(node)){
                if(!visited_[node]){
                    labels_[node] = numberOfLabels_;
                    queue_.push(node);
                    visited_[node] = true;
                    while(!queue_.empty()){
                        
                        const auto w = queue_.front();
                        queue_.pop();

                        for(const auto adj : graph_.adjacency(w)){
                            const auto n = adj.node();
                            const auto e = adj.edge();
                            if(!visited_[n] && mask.useNode(n) && mask.useEdge(e)){
                                labels_[n] = numberOfLabels_;
                                queue_.push(n);
                                visited_[n] = true;
                            }
                        }

                    }
                    ++numberOfLabels_;
                }
            }
        });

        needsReset_ = true;

        return numberOfLabels_;

    }

    void reset(){
        numberOfLabels_ = 0;
        for(const auto node : graph_.nodes()){
            visited_[node] = false;
        }

        while(!queue_.empty()){
            queue_.pop();
        }
    }

    bool areConnected(const int64_t u, const int64_t v) const{
        return labels_[u] == labels_[v];
    }

    uint64_t componentLabel(const uint64_t u){
        return labels_[u];
    }

    uint64_t operator[](const uint64_t u) const {
        return labels_[u];
    }
    
    uint64_t maxLabel()const{
        return numberOfLabels_ - 1;
    }



private:
    const GraphType & graph_;
    LabelsMapType labels_;
    VisitedMapType visited_;
    uint64_t numberOfLabels_;
    bool needsReset_;
    std::queue<uint64_t> queue_;
};












} // namespace nifty::graph
} // namespace nifty

