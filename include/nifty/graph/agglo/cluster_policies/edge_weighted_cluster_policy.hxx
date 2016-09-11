#pragma once
#ifndef NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX
#define NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX

#include <functional>



#include "vigra/priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"

namespace nifty{
namespace graph{
namespace agglo{







template<
    class GRAPH,class EDGE_INDICATORS,
    class EDGE_SIZES,class NODE_SIZES,
    bool ENABLE_UCM
>
class EdgeWeightedClusterPolicy{

    typedef EdgeWeightedClusterPolicy<
        GRAPH, EDGE_INDICATORS, 
        EDGE_SIZES, NODE_SIZES,
        ENABLE_UCM
    > SelfType;


public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef EDGE_INDICATORS                             EdgeIndicatorsType;
    typedef EDGE_SIZES                                  EdgeSizesType;
    typedef NODE_SIZES                                  NodeSizesType;
    typedef EdgeWeightedClusterPolicySettings           Settings;
    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types
    
    typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    EdgeWeightedClusterPolicy(const GraphType &, EdgeIndicatorsType, EdgeSizesType, 
                              NodeSizesType,const Settings & settings = Settings());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone() const;

    // callback called by edge contraction graph
    
    EdgeContractionGraphType & edgeContractionGraph();

private:
    void initializeWeights() ;
    double computeWeight(const uint64_t edge) const;

public:
    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);

private:
    // INPUT
    const GraphType &   graph_;
    EdgeIndicatorsType  edgeIndicators_;
    EdgeSizesType       edgeSizes_;
    NodeSizesType       nodeSizes_;
    Settings            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

};


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
EdgeWeightedClusterPolicy(
    const GraphType & graph,
    EdgeIndicatorsType  edgeIndicators,
    EdgeSizesType       edgeSizes,
    NodeSizesType       nodeSizes,
    const Settings & settings
)
:   graph_(graph),
    edgeIndicators_(edgeIndicators),
    edgeSizes_(edgeSizes),
    nodeSizes_(nodeSizes),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
    this->initializeWeights();
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
edgeToContractNext() const {
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline bool 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
isDone() const {
    if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop)
        return  true;
    if(edgeContractionGraph_.numberOfEdges() <= settings_.numberOfEdgesStop)
        return  true;
    return false;
}


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
initializeWeights() {
    for(const auto edge : graph_.edges())
        pq_.push(edge, this->computeWeight(edge));
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline double 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
computeWeight(
    const uint64_t edge
) const {
    const auto sr = settings_.sizeRegularizer;
    const auto uv = edgeContractionGraph_.uv(edge);
    const auto sizeU = nodeSizes_[uv.first];
    const auto sizeV = nodeSizes_[uv.second];
    const auto sFac = 2.0 / ( 1.0/std::pow(sizeU,sr) + 1.0/std::pow(sizeV,sr) );
    return edgeIndicators_[edge] * sFac;
}


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline typename EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::EdgeContractionGraphType & 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    nodeSizes_[aliveNode] +=nodeSizes_[deadNode];
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    pq_.deleteItem(deadEdge);
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];
    const auto s = sa + sd;
    edgeIndicators_[aliveEdge] = (sa*edgeIndicators_[aliveEdge] + sd*edgeIndicators_[deadEdge])/s;
    edgeSizes_[aliveEdge] = s;
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES, bool ENABLE_UCM>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    // HERE WE UPDATE 
    const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
    for(auto adj : edgeContractionGraph_.adjacency(u)){
        const auto edge = adj.edge();
        pq_.push(edge, computeWeight(edge));
    }
}




} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX*/
