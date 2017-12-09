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
class FixationClusterPolicy3{

    typedef FixationClusterPolicy3<
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

    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_MERGE_EDGE, class EDGE_SIZES>
    FixationClusterPolicy3(const GraphType &, 
                              const MERGE_PRIOS & , 
                              const NOT_MERGE_PRIOS &,
                              const IS_MERGE_EDGE &,
                              const EDGE_SIZES & , 
                              const SettingsType & settings = SettingsType());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone();

    // callback called by edge contraction graph
    
    EdgeContractionGraphType & edgeContractionGraph();

private:
    double pqActionPrio(const uint64_t edge) const;

public:
    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);

    bool isMergeAllowed(const uint64_t edge){
        return !isNonLinkEdge_[edge] && isMergeEdge_[edge];
    }
    void addNonLinkConstraint(const uint64_t edge){
        const auto uv = edgeContractionGraph_.uv(edge);
        std::cout<<"add non link constraint on edge "<<edge<<" "<<uv.first<<" "<<uv.second<<"\n";
        isNonLinkEdge_[edge] = 1;
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



    EdgePrioType mergePrios_;
    EdgePrioType notMergePrios_; 

    UInt8EdgeMap isMergeEdge_;


    EdgeSizesType       edgeSizes_;
    SettingsType        settings_;
    
    // INTERNAL
    UInt8EdgeMap isNonLinkEdge_;
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;


    uint64_t edgeToContractNext_;
    double   edgeToContractNextMergePrio_;
};


template<class GRAPH, bool ENABLE_UCM>
template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_MERGE_EDGE,class EDGE_SIZES>
inline FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
FixationClusterPolicy3(
    const GraphType & graph,
    const MERGE_PRIOS & mergePrios,
    const NOT_MERGE_PRIOS & notMergePrios,
    const IS_MERGE_EDGE & isMergeEdge,
    const EDGE_SIZES      & edgeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    mergePrios_(graph),
    notMergePrios_(graph),
    isMergeEdge_(graph),
    edgeSizes_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    isNonLinkEdge_(graph, 0),
    edgeContractionGraph_(graph, *this)
{
    //std::cout<<"constructor\n";
    graph_.forEachEdge([&](const uint64_t edge){
        const auto uv = edgeContractionGraph_.uv(edge);
        std::cout<<"uv "<<uv.first<<" "<<uv.second<<"\n";
        mergePrios_[edge] = mergePrios[edge];
        notMergePrios_[edge] = notMergePrios[edge];
        isMergeEdge_[edge] = isMergeEdge[edge]; 

        if(isMergeEdge_[edge]){
            notMergePrios_[edge] = 0;
        }
        else{
            mergePrios_[edge] = 0;
        }

        edgeSizes_[edge] = edgeSizes[edge];
        pq_.push(edge, this->pqActionPrio(edge));


        std::cout<<" mp"<<mergePrios_[edge]<<" nmp "<<notMergePrios_[edge]<< " "<<pqActionPrio(edge)<<"\n";

    });
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {    
    return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
isDone()     {
    if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop || pq_.empty() || pq_.topPriority() < -0.0000001){
        return  true;
    }
    else{
        while(pq_.topPriority() > -0.0000001 ){
            const auto nextActioneEdge = pq_.top();
            const auto uv = edgeContractionGraph_.uv(nextActioneEdge);

            std::cout<<"PRIO: "<<pq_.topPriority()<<"next action edge "<<uv.first<<" "<<uv.second<<"\n";
            if(isMergeEdge_[nextActioneEdge]){
                if(this->isMergeAllowed(nextActioneEdge)){
                    edgeToContractNext_ = nextActioneEdge;
                    edgeToContractNextMergePrio_ = pq_.topPriority();
                    return false;
                }
                else{
                    this->addNonLinkConstraint(nextActioneEdge);
                    pq_.push(nextActioneEdge, -1.0);
                }
            }
            else{
                this->addNonLinkConstraint(nextActioneEdge);
                pq_.push(nextActioneEdge, -1.0);
            }
        }
        return true;
    }
}


template<class GRAPH, bool ENABLE_UCM>
inline double 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
pqActionPrio(
    const uint64_t edge
) const {

    if(isMergeEdge_[edge]){
        return mergePrios_[edge];//+ 0.1*float(nu+nv);
    }
    else{
        return notMergePrios_[edge];
    }
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    std::cout<<"contract edge: "<<edgeToContract<<"\n"; 
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename FixationClusterPolicy3<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    std::cout<<"    merge nodes: a/d "<<aliveNode<<" "<<deadNode<<" \n"; 
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    std::cout<<"    merge edges: a/d "<<aliveEdge<<" "<<deadEdge<<" \n"; 

    NIFTY_CHECK_OP(aliveEdge,!=,deadEdge,"");
    NIFTY_CHECK(pq_.contains(aliveEdge),"");
    NIFTY_CHECK(pq_.contains(deadEdge),"");
    
    pq_.deleteItem(deadEdge);
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];
    const auto s = sa + sd;


    const auto deadIsMergeEdge = isMergeEdge_[deadEdge];
    auto & aliveIsMergeEdge = isMergeEdge_[aliveEdge];
    
    mergePrios_[aliveEdge]    = std::max(mergePrios_[aliveEdge]    , mergePrios_[deadEdge]);
    notMergePrios_[aliveEdge] = std::max(notMergePrios_[aliveEdge] , notMergePrios_[deadEdge]);
    
    isNonLinkEdge_[aliveEdge] = std::max(isNonLinkEdge_[aliveEdge], isNonLinkEdge_[deadEdge]);


    if(deadIsMergeEdge != aliveIsMergeEdge){
        aliveIsMergeEdge = mergePrios_[aliveEdge] >= notMergePrios_[aliveEdge];
    }

    edgeSizes_[aliveEdge] = s;
    
    // update prios
    pq_.push(aliveEdge, this->pqActionPrio(aliveEdge));
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
FixationClusterPolicy3<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    std::cout<<"contract edge done: "<<edgeToContract<<"\n\n";

}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

