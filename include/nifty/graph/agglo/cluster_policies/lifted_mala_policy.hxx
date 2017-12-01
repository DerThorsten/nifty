#pragma once

#include <functional>



#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"

namespace nifty{
namespace graph{
namespace agglo{







template<
    class GRAPH,bool ENABLE_UCM
>
class LiftedMalaPolicy{

    typedef LiftedMalaPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

private:
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;

public:
    // input types
    typedef GRAPH                                        GraphType;
    typedef FloatEdgeMap                                 EdgeIndicatorsType;
    typedef FloatEdgeMap                                 EdgeSizesType;
    typedef UInt8EdgeMap                                 EdgeIsLiftedType;
    typedef FloatNodeMap                                 NodeSizesType;

    struct SettingsType{

    };
    typedef EdgeContractionGraph<GraphType, SelfType>    EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types
    // internal types
    const static size_t NumberOfBins = 20;
    typedef std::array<float, NumberOfBins> HistogramType;     

    typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    template<
        class EDGE_INDICATORS, 
        class EDGE_SIZES,
        class IS_LIFTED_EDGES,
        class NODE_SIZES
    >
    LiftedMalaPolicy(const GraphType &, 
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const IS_LIFTED_EDGES &,
                              const NODE_SIZES & ,
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
    EdgeIsLiftedType    isLiftedEdge_;
    NodeSizesType       nodeSizes_;
    SettingsType            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

};


template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS, class EDGE_SIZES,class IS_LIFTED_EDGES, class NODE_SIZES>
inline LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
LiftedMalaPolicy(
    const GraphType & graph,
    const EDGE_INDICATORS & edgeIndicators,
    const EDGE_SIZES      & edgeSizes,
    const IS_LIFTED_EDGES & isLiftedEdge,
    const NODE_SIZES      & nodeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    edgeIndicators_(graph),
    edgeSizes_(graph),
    isLiftedEdge_(graph),
    nodeSizes_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
    graph_.forEachEdge([&](const uint64_t edge){
        edgeIndicators_[edge] = edgeIndicators[edge];
        edgeSizes_[edge] = edgeSizes[edge];
        isLiftedEdge_[edge] = isLiftedEdge[edge];
    });
    graph_.forEachNode([&](const uint64_t node){
        nodeSizes_[node] = nodeSizes[node];
    });
    this->initializeWeights();
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {
    const auto edgeToContract = pq_.top();
    NIFTY_CHECK(!isLiftedEdge_[edgeToContract], "internal error");
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
isDone() const {
    if(pq_.topPriority() > 0.5){
        return true;
    }
    else{
        return pq_.empty();
    }
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
initializeWeights() {
    for(const auto edge : graph_.edges())
        pq_.push(edge, this->computeWeight(edge));
}

template<class GRAPH, bool ENABLE_UCM>
inline double 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
computeWeight(
    const uint64_t edge
) const {
    if(isLiftedEdge_[edge]){
        return std::numeric_limits<float>::infinity();
    }
    else{
        return edgeIndicators_[edge];// * sFac;
    }
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename LiftedMalaPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    nodeSizes_[aliveNode] +=nodeSizes_[deadNode];
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    pq_.deleteItem(deadEdge);
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];
    const auto s = sa + sd;


    auto & la = isLiftedEdge_[aliveEdge];
    const auto & ld = isLiftedEdge_[deadEdge];

    la = la && ld;


    edgeIndicators_[aliveEdge] = (sa*edgeIndicators_[aliveEdge] + sd*edgeIndicators_[deadEdge])/s;
    edgeSizes_[aliveEdge] = s;

    pq_.push(aliveEdge, computeWeight(aliveEdge));
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
LiftedMalaPolicy<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    //// HERE WE UPDATE 
    //const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
    //for(auto adj : edgeContractionGraph_.adjacency(u)){
    //    const auto edge = adj.edge();
    //    pq_.push(edge, computeWeight(edge));
    //}
}




} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

