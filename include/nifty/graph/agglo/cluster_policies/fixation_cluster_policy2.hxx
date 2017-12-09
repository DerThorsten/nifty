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
    class GRAPH,bool ENABLE_UCM
>
class FixationClusterPolicy2{

    typedef FixationClusterPolicy2<
        GRAPH, ENABLE_UCM
    > SelfType;

private:    
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;


    typedef boost::container::flat_set<uint64_t> SetType;
    //typedef std::set<uint64_t> SetType;
    //typedef std::unordered_set<uint64_t> SetType;

    typedef typename GRAPH:: template NodeMap<SetType > NonLinkConstraints;


public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgePrioType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;

    struct SettingsType{
        uint64_t numberOfNodesStop{1};
    };


    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types


    typedef nifty::tools::ChangeablePriorityQueue< double , std::greater<double> > QueueType;

public:

    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE, class EDGE_SIZES>
    FixationClusterPolicy2(const GraphType &, 
                              const MERGE_PRIOS & , 
                              const NOT_MERGE_PRIOS &,
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

    bool isMergeAllowed(const uint64_t edge){
        if(isLocalEdge_[edge]){
           return isPureLocal_[edge] ? true : mergePrios_[edge] > notMergePrios_[edge];
        }
        else{
            return false;
        }
    }
    

    const EdgePrioType & mergePrios() const {
        return mergePrios_;
    }
    const EdgePrioType & notMergePrios() const {
        return notMergePrios_;
    }
    const EdgeSizesType & edgeSizes() const {
        return edgeSizes_;
    }

    
private:
    // INPUT
    const GraphType &   graph_;


    NonLinkConstraints nonLinkConstraints_;

    EdgePrioType mergePrios_;
    EdgePrioType notMergePrios_; 

    UInt8EdgeMap isLocalEdge_;
    UInt8EdgeMap isPureLocal_;

    EdgeSizesType       edgeSizes_;
    SettingsType        settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;


    uint64_t edgeToContractNext_;
    double   edgeToContractNextMergePrio_;
};


template<class GRAPH, bool ENABLE_UCM>
template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE,class EDGE_SIZES>
inline FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
FixationClusterPolicy2(
    const GraphType & graph,
    const MERGE_PRIOS & mergePrios,
    const NOT_MERGE_PRIOS & notMergePrios,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    nonLinkConstraints_(graph),
    mergePrios_(graph),
    notMergePrios_(graph),
    isLocalEdge_(graph),
    isPureLocal_(graph),
    edgeSizes_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
    //std::cout<<"constructor\n";
    graph_.forEachEdge([&](const uint64_t edge){
        isLocalEdge_[edge] = isLocalEdge[edge];

        if(isLocalEdge_[edge]){
            notMergePrios_[edge] = 0.0;
            mergePrios_[edge] = mergePrios[edge];
        }
        else{
            notMergePrios_[edge] = notMergePrios[edge];
            mergePrios_[edge] = 0.0;
        }
        isPureLocal_[edge] = isLocalEdge[edge];
        edgeSizes_[edge] = edgeSizes[edge];
        pq_.push(edge, this->pqMergePrio(edge));
    });
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {    
    return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
isDone()     {
    if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop){
        //std::cout<<"done a1\n";
        return  true;
    }
    else if(pq_.empty() || pq_.topPriority() <  -0.0000001){
        //std::cout<<"done a2\n";
        return  true;
    }
    else{
        while(pq_.topPriority() > -0.0000001 ){
            const auto nextActioneEdge = pq_.top();
            if(isLocalEdge_[nextActioneEdge]){
                if(this->isMergeAllowed(nextActioneEdge)){
                    edgeToContractNext_ = nextActioneEdge;
                    edgeToContractNextMergePrio_ = pq_.topPriority();
                    //std::cout<<"not done\n";
                    return false;
                }
                else{
                    pq_.push(nextActioneEdge, -1.0);
                }
            }
            else{
                pq_.push(nextActioneEdge, -1.0);
            }
        }
        //std::cout<<"done b\n";
        return true;
    }
}


template<class GRAPH, bool ENABLE_UCM>
inline double 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
pqMergePrio(
    const uint64_t edge
) const {
    return isLocalEdge_[edge] ?  double(mergePrios_[edge]) : -1.0; 
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    std::cout<<"contract edge: "<<edgeToContract<<"\n"; 
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename FixationClusterPolicy2<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    std::cout<<"    merge nodes: a/d "<<aliveNode<<" "<<deadNode<<" \n"; 
    /*
    auto  & aliveNodeNlc = nonLinkConstraints_[aliveNode];
    const auto & deadNodeNlc = nonLinkConstraints_[deadNode];
    aliveNodeNlc.insert(deadNodeNlc.begin(), deadNodeNlc.end());
    for(const auto v : deadNodeNlc){
        auto & nlc = nonLinkConstraints_[v];

        // best way to change values in set... 
        nlc.erase(deadNode);
        nlc.insert(aliveNode);
    }
    aliveNodeNlc.erase(deadNode);
    */

}

template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    std::cout<<"    merge edges: a/d "<<aliveEdge<<" "<<deadEdge<<" \n"; 
    NIFTY_CHECK_OP(aliveEdge,!=,deadEdge,"");
    NIFTY_CHECK(pq_.contains(aliveEdge),"");
    NIFTY_CHECK(pq_.contains(deadEdge),"");
    
    isPureLocal_[aliveEdge] = isPureLocal_[aliveEdge] && isPureLocal_[deadEdge];
    pq_.deleteItem(deadEdge);
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];
    const auto s = sa + sd;


    const auto deadIsLocalEdge = isLocalEdge_[deadEdge];
    auto & aliveIsLocalEdge = isLocalEdge_[aliveEdge];
    
    aliveIsLocalEdge = deadIsLocalEdge || aliveIsLocalEdge;

    mergePrios_[aliveEdge]    = std::max(mergePrios_[aliveEdge]    , mergePrios_[deadEdge]);
    notMergePrios_[aliveEdge] = std::max(notMergePrios_[aliveEdge] , notMergePrios_[deadEdge]);
       
    
    // update prios
    
    pq_.push(aliveEdge, this->pqMergePrio(aliveEdge));
    
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy2<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    std::cout<<"contract edge done: "<<edgeToContract<<"\n\n";
}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

