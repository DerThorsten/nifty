#pragma once
#ifndef NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX
#define NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX

#include <functional>

#include "vigra/priority_queue.hxx"

#include "nifty/graph/edge_contraction_graph.hxx"

namespace nifty{
namespace graph{
namespace agglo{



struct EdgeWeightedClusterPolicySettings{
    double sizeRegularizer{0.5};
    uint64_t numberOfNodesStop{1};
};



template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
class EdgeWeightedClusterPolicy{

    typedef EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES> SelfType;
public:
    // input types
    typedef GRAPH                               GraphType;
    typedef EDGE_INDICATORS                     EdgeIndicatorsType;
    typedef EDGE_SIZES                          EdgeSizesType;
    typedef NODE_SIZES                          NodeSizesType;
    typedef EdgeWeightedClusterPolicySettings   Settings;
private:

    // internal types
    typedef EdgeContractionGraph<GraphType, SelfType> EdgeContractionGraphType;
    typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    EdgeWeightedClusterPolicy(const GraphType &, EdgeIndicatorsType, EdgeSizesType, 
                              NodeSizesType,const Settings & settings = Settings());

    uint64_t edgeToContractNext() const;
    bool isDone() const;


private:
    void initializeWeights() const;
    double computeWeight(const uint64_t edge) const;

    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);


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


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
EdgeWeightedClusterPolicy(
    const GraphType & graph,
    EdgeIndicatorsType  edgeIndicators,
    EdgeSizesType       edgeSizes,
    NodeSizesType       nodeSizes,
    const Settings & settings
)
:
graph_(graph),
edgeIndicators_(edgeIndicators),
edgeSizes_(edgeSizes),
nodeSizes_(nodeSizes),
pq_(graph.edgeIdUpperBound()+1),
settings_(settings),
edgeContractionGraph_(graph, *this){
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline uint64_t 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
edgeToContractNext() const {
    return pq_.top();
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline bool 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
isDone() const {
    return edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop;
}


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
initializeWeights() const {
    for(const auto edge : graph_.edges())
        pq_.push(edge, this->computeWeight(edge));
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline double 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
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


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    nodeSizes_[aliveNode] +=nodeSizes_[deadNode];
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
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

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES>
inline void 
EdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    // HERE WE UPDATE 
    const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
    for(auto adj : edgeContractionGraph_.adjacency(u)){
        const auto edge = adj.edge();
        //const auto p = this->recomputeFeaturesAndPredictImpl(edge);
    }
}




} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX*/
