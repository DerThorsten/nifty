#pragma once

#include <functional>
#include <set>
#include <unordered_set>
#include <boost/container/flat_set.hpp>

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"



namespace nifty{
namespace graph{
namespace agglo{


template<
    class GRAPH, class ACC, bool ENABLE_UCM
>
class LiftedGraphEdgeWeightedClusterPolicy{

    typedef LiftedGraphEdgeWeightedClusterPolicy<
        GRAPH, ACC,  ENABLE_UCM
    > SelfType;

private:    
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;


    typedef ACC AccType;

public:
    typedef typename AccType::SettingsType AccSettingsType;



    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgePrioType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;

    struct SettingsType{

     
        AccSettingsType updateRule;
        uint64_t numberOfNodesStop{1};
        double stopPriority{0.5};
    };


    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types


    typedef nifty::tools::ChangeablePriorityQueue< double , std::greater<double> > QueueType;

public:

    template<class MERGE_PRIOS, class IS_LOCAL_EDGE, class EDGE_SIZES>
    LiftedGraphEdgeWeightedClusterPolicy(const GraphType &, 
                              const MERGE_PRIOS & , 
                              const IS_LOCAL_EDGE &,
                              const EDGE_SIZES & , 
                              const SettingsType & settings = SettingsType());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone();

    // callback called by edge contraction graph
    
    EdgeContractionGraphType & edgeContractionGraph();

private:
    double pqMergePrio(const uint64_t edge) const;

public:
    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);

   
private:



    // INPUT
    const GraphType &   graph_;


    ACC acc_;


    UInt8EdgeMap isLocalEdge_;
    UInt8EdgeMap isPureLocal_;
    UInt8EdgeMap isPureLifted_;
    SettingsType        settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

    uint64_t edgeToContractNext_;
    double   edgeToContractNextMergePrio_;
};


template<class GRAPH, class ACC, bool ENABLE_UCM>
template<class MERGE_PRIOS, class IS_LOCAL_EDGE,class EDGE_SIZES>
inline LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
LiftedGraphEdgeWeightedClusterPolicy(
    const GraphType & graph,
    const MERGE_PRIOS & mergePrios,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    acc_(graph, mergePrios,    edgeSizes, settings.updateRule),
    isLocalEdge_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
   
    graph_.forEachEdge([&](const uint64_t edge){
        const auto loc = isLocalEdge[edge];
        isLocalEdge_[edge] = loc;
        pq_.push(edge, this->pqMergePrio(edge));
    });
}

template<class GRAPH, class ACC, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
edgeToContractNext() const {    
    return std::pair<uint64_t, double>(pq_.top(), pq_.topPriority()) ;
}

template<class GRAPH, class ACC, bool ENABLE_UCM>
inline bool 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::isDone(){
    return edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop ||
       pq_.empty() || pq_.topPriority() < settings_.stopPriority;
}

template<class GRAPH, class ACC, bool ENABLE_UCM>
inline double 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
pqMergePrio(
    const uint64_t edge
) const {
    return isLocalEdge_[edge] ?  acc_[edge] : -1.0*std::numeric_limits<double>::infinity(); 
}

template<class GRAPH, class ACC, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, class ACC, bool ENABLE_UCM>
inline typename LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::EdgeContractionGraphType & 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, class ACC, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
}

template<class GRAPH, class ACC, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){

    NIFTY_ASSERT_OP(aliveEdge,!=,deadEdge);
    NIFTY_ASSERT(pq_.contains(aliveEdge));
    NIFTY_ASSERT(pq_.contains(deadEdge));

    pq_.deleteItem(deadEdge);
   
    // update merge prio

    acc_.merge(aliveEdge, deadEdge);

    



    const auto deadIsLocalEdge = isLocalEdge_[deadEdge];
    auto & aliveIsLocalEdge = isLocalEdge_[aliveEdge];
    aliveIsLocalEdge = deadIsLocalEdge || aliveIsLocalEdge;


    pq_.push(aliveEdge, this->pqMergePrio(aliveEdge));
    
}


template<class GRAPH, class ACC, bool ENABLE_UCM>
inline void 
LiftedGraphEdgeWeightedClusterPolicy<GRAPH, ACC, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    
}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

