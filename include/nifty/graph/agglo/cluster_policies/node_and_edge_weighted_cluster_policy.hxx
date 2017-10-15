#pragma once

#include <functional>



#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"

namespace nifty{
namespace graph{
namespace agglo{







template<
    class GRAPH,bool ENABLE_UCM
>
class NodeAndEdgeWeightedClusterPolicy{

    typedef NodeAndEdgeWeightedClusterPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

public:
    typedef GRAPH                                     GraphType;

    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;

    typedef nifty::graph::graph_maps::MultibandNodeMap<GraphType, double> NodeFeatureMap;

    
    typedef FloatEdgeMap                                EdgeIndicatorsType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;
   


    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:
    struct SettingsType{
        double beta{0.5};
        double sizeRegularizer{0.5};
        uint64_t numberOfNodesStop{1};
        uint64_t numberOfEdgesStop{0};
    };

    template<
        class EDGE_INDICATORS, 
        class EDGE_SIZES, 
        class NODE_FEATURES,
        class NODE_SIZES
    >
    NodeAndEdgeWeightedClusterPolicy(const GraphType &, 
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const NODE_FEATURES &,
                              const NODE_SIZES & ,
                              const SettingsType & settings = SettingsType());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone() const;

    // callback called by edge contraction graph
    
    EdgeContractionGraphType & edgeContractionGraph();

private:
    void initializeWeights() ;
    double computeWeight(const uint64_t edge) const;
    double weightFromNodes(const uint64_t u, const uint64_t v) const;

public:
    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);


    const EdgeIndicatorsType & edgeIndicators() const {
        return edgeIndicators_;
    }
    const EdgeSizesType & edgeSizes() const {
        return edgeSizes_;
    }
    const NodeSizesType & nodeSizes() const {
        return nodeSizes_;
    }
    
private:
    // INPUT
    const GraphType &   graph_;
    EdgeIndicatorsType  edgeIndicators_;
    EdgeSizesType       edgeSizes_;
    NodeFeatureMap      nodeFeatures_;
    NodeSizesType       nodeSizes_;
    SettingsType            settings_;
    
    // INTERNAL
    const uint64_t nChannels_;
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

};


template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_FEATURES, class NODE_SIZES>
inline NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
NodeAndEdgeWeightedClusterPolicy(
    const GraphType & graph,
    const EDGE_INDICATORS & edgeIndicators,
    const EDGE_SIZES      & edgeSizes,
    const NODE_FEATURES   & nodeFeatures,
    const NODE_SIZES      & nodeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    edgeIndicators_(graph),
    edgeSizes_(graph),
    nodeFeatures_(graph_, nodeFeatures.numberOfChannels()),
    nodeSizes_(graph),
    settings_(settings),
    nChannels_(nodeFeatures.numberOfChannels()),
    edgeContractionGraph_(graph, *this),
    pq_(graph.edgeIdUpperBound()+1)
{
    graph_.forEachEdge([&](const uint64_t edge){
        edgeIndicators_[edge] = edgeIndicators[edge];
        edgeSizes_[edge] = edgeSizes[edge];
    });

    const auto  nChannels = nodeFeatures.numberOfChannels();
    graph_.forEachNode([&](const uint64_t node){
        nodeSizes_[node] = nodeSizes[node];
        auto valProxy = nodeFeatures_[node];
        const auto valProxyIn = nodeFeatures[node];
        for(auto c=0; c<nChannels; ++c){
            valProxy[c] = valProxyIn[c];
        }
    });
    this->initializeWeights();
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
isDone() const {
    if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop)
        return  true;
    if(edgeContractionGraph_.numberOfEdges() <= settings_.numberOfEdgesStop)
        return  true;
    return false;
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
initializeWeights() {
    for(const auto edge : graph_.edges())
        pq_.push(edge, this->computeWeight(edge));
}

template<class GRAPH, bool ENABLE_UCM>
inline double 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
computeWeight(
    const uint64_t edge
) const {
    const auto sr = settings_.sizeRegularizer;
    const auto uv = edgeContractionGraph_.uv(edge);
    const auto sizeU = nodeSizes_[uv.first];
    const auto sizeV = nodeSizes_[uv.second];
    const auto sFac = 2.0 / ( 1.0/std::pow(sizeU,sr) + 1.0/std::pow(sizeV,sr) );
    const auto fromNodes = this->weightFromNodes(uv.first, uv.second);
    const auto fromEdge = edgeIndicators_[edge];
    const auto beta = settings_.beta;
    const auto e = beta*fromNodes + (1.0-beta)*fromEdge;
    return e * sFac;
}
    

template<class GRAPH, bool ENABLE_UCM>
inline double 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
weightFromNodes(
    const uint64_t u, const uint64_t v
) const {
    const auto featU = nodeFeatures_[u];
    const auto featV = nodeFeatures_[v];
    double d = 0;
    for(auto c=0; c<nChannels_; ++c){
        const auto dd =  std::abs(featU[c] - featV[c]);
        d += dd*dd;
    }
    return std::sqrt(d);
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    // proxy object, no plain return by value !
    auto       featA = nodeFeatures_[aliveNode];
    const auto featD = nodeFeatures_[deadNode];

    const auto sizeA = nodeSizes_[aliveNode];
    const auto sizeD = nodeSizes_[deadNode];
    const auto size = sizeA + sizeD;
    for(auto c=0; c<nChannels_; ++c){
        featA[c] = (sizeA*featA[c] + sizeD*featD[c])/size;
    }
    nodeSizes_[aliveNode] +=nodeSizes_[deadNode];
}   

template<class GRAPH, bool ENABLE_UCM>
inline void 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
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

template<class GRAPH, bool ENABLE_UCM>
inline void 
NodeAndEdgeWeightedClusterPolicy<GRAPH, ENABLE_UCM>::
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

