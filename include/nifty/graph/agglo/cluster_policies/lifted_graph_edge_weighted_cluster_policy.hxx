#pragma once

#include <functional>

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"

namespace nifty{
namespace graph{
namespace agglo{







template<
    class GRAPH, class EDGE_INDICATORS,
    class EDGE_SIZES, class NODE_SIZES,
    class EDGE_IS_LIFTED, bool ENABLE_UCM = true
>
class LiftedGraphEdgeWeightedClusterPolicy{

    typedef LiftedGraphEdgeWeightedClusterPolicy<
        GRAPH, EDGE_INDICATORS, 
        EDGE_SIZES, NODE_SIZES, 
        EDGE_IS_LIFTED, ENABLE_UCM
    > SelfType;


public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef EDGE_INDICATORS                             EdgeIndicatorsType;
    typedef EDGE_SIZES                                  EdgeSizesType;
    typedef NODE_SIZES                                  NodeSizesType;
    typedef EDGE_IS_LIFTED                              EdgeIsLifted;
    typedef EdgeWeightedClusterPolicySettings           SettingsType;
    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;


    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    typedef typename GraphType:: template EdgeMap<double> CurrentWeightMap;

    // internal types
    
    typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    LiftedGraphEdgeWeightedClusterPolicy(
        const GraphType &, EdgeIndicatorsType, 
        EdgeSizesType, NodeSizesType,
        EdgeIsLifted,
        const SettingsType & settings = SettingsType());

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
    EdgeIsLifted        edgeIsLifted_;
    SettingsType            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;
    CurrentWeightMap currentWeight_;

};


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
LiftedGraphEdgeWeightedClusterPolicy(
    const GraphType & graph,
    EdgeIndicatorsType  edgeIndicators,
    EdgeSizesType       edgeSizes,
    NodeSizesType       nodeSizes,
    EdgeIsLifted        edgeIsLifted,
    const SettingsType & settings
)
:   graph_(graph),
    edgeIndicators_(edgeIndicators),
    edgeSizes_(edgeSizes),
    nodeSizes_(nodeSizes),
    settings_(settings),
    edgeIsLifted_(edgeIsLifted),
    edgeContractionGraph_(graph, *this),
    pq_(graph.edgeIdUpperBound()+1)  
    //currentWeight_(graph)
{
    this->initializeWeights();
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
edgeToContractNext() const {
    const auto edgeToContract = pq_.top();
    NIFTY_CHECK(!edgeIsLifted_[edgeToContract], "internal error");
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline bool 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
isDone() const {
    if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop)
        return  true;
    if(edgeContractionGraph_.numberOfEdges() <= settings_.numberOfEdgesStop)
        return  true;
    return false;
}


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
initializeWeights() {
    for(const auto edge : graph_.edges()){

        if(edgeIsLifted_[edge]){
            pq_.push(edge, std::numeric_limits<double>::infinity() );
        }
        else{
            const auto w  =  this->computeWeight(edge);
            pq_.push(edge, w);
        }
        
    }
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline double 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
computeWeight(
    const uint64_t edge
) const {

    if(!edgeIsLifted_[edge]){

        const auto sr = settings_.sizeRegularizer;
        const auto uv = edgeContractionGraph_.uv(edge);
        const auto sizeU = nodeSizes_[uv.first];
        const auto sizeV = nodeSizes_[uv.second];
        const auto sFac = 2.0 / ( 1.0/std::pow(sizeU,sr) + 1.0/std::pow(sizeV,sr) );
        return edgeIndicators_[edge] * sFac;
    }
    else{
        NIFTY_CHECK(false, "internal error");
    }
}


template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    NIFTY_CHECK(!edgeIsLifted_[edgeToContract], "internal error");
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline typename LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::EdgeContractionGraphType & 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    nodeSizes_[aliveNode] += nodeSizes_[deadNode];
}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    pq_.deleteItem(deadEdge);
    
    
    const auto aIsLifted = edgeIsLifted_[aliveEdge];
    const auto dIsLifted = edgeIsLifted_[deadEdge];



    // non is lifted, merge as always
    if(!aIsLifted && ! dIsLifted){

        const auto sa = edgeSizes_[aliveEdge];
        const auto sd = edgeSizes_[deadEdge];
        const auto s = sa + sd;

        edgeIndicators_[aliveEdge] = (sa*edgeIndicators_[aliveEdge] + sd*edgeIndicators_[deadEdge])/s;
        edgeSizes_[aliveEdge] = s;
    }
    // both are lifted => merge but keep pq weight at -inf
    else if(aIsLifted &&  dIsLifted){
        // no change
    }
    // if only the dead edge is lifted
    // we need can merge as always
    else if(!aIsLifted && dIsLifted){
        // no change
    }
    // alive edge was lifted, but merged with non lifted
    // which makes the lifted edge a normal edge
    else if(aIsLifted && !dIsLifted){
        edgeSizes_[aliveEdge] = edgeSizes_[deadEdge];
        edgeIndicators_[aliveEdge] = edgeIndicators_[deadEdge];
        edgeIsLifted_[aliveEdge] = false;
    }
    else{
        NIFTY_CHECK(false,"bug");
    }




}

template<class GRAPH,class EDGE_INDICATORS,class EDGE_SIZES,class NODE_SIZES,class EDGE_IS_LIFTED, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, EDGE_INDICATORS, EDGE_SIZES, NODE_SIZES, EDGE_IS_LIFTED, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    // HERE WE UPDATE 
    const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
    for(auto adj : edgeContractionGraph_.adjacency(u)){
        const auto edge = adj.edge();
        if(!edgeIsLifted_[edge]){
            pq_.push(edge, computeWeight(edge));
        }
    }
}




} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

