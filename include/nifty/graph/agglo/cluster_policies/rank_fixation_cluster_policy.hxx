#pragma once

#include <functional>
#include <set>
#include <unordered_set>
#include <boost/container/flat_set.hpp>

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"
#include "nifty/histogram/histogram.hxx"


namespace nifty{
namespace graph{
namespace agglo{


template<
    class GRAPH,bool ENABLE_UCM
>
class RankFixationClusterPolicy{

    typedef RankFixationClusterPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

private:    


    typedef nifty::histogram::Histogram<float> HistogramType;

    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template EdgeMap<HistogramType> HistEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;


    typedef boost::container::flat_set<uint64_t> SetType;
    //typedef std::set<uint64_t> SetType;
    //typedef std::unordered_set<uint64_t> SetType;




public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgePrioType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;

    struct SettingsType{
        bool zeroInit = false;
        double q0{0.5};
        double q1{0.5};
        uint64_t numberOfBins{40};
        uint64_t numberOfNodesStop{1};
    };


    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types


    typedef nifty::tools::ChangeablePriorityQueue< double , std::greater<double> > QueueType;

public:

    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE, class EDGE_SIZES>
    RankFixationClusterPolicy(const GraphType &, 
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
            // todo this isPureLocal_ seems to be legacy
            // check if needed
           return isPureLocal_[edge] ? true :getMergePrio(edge)> getNotMergePrio(edge);
        }
        else{
            return false;
        }
    }

    const EdgeSizesType & edgeSizes() const {
        return edgeSizes_;
    }

    
private:

    const double getMergePrio(const uint64_t edge)const{
        return mergePriosHist_[edge].rank(settings_.q0);
    }

    const double getNotMergePrio(const uint64_t edge)const{
        return notMergePriosHist_[edge].rank(settings_.q1);
    }

    // INPUT
    const GraphType &   graph_;



    HistEdgeMap mergePriosHist_;
    HistEdgeMap notMergePriosHist_; 

    UInt8EdgeMap isLocalEdge_;
    UInt8EdgeMap isPureLocal_;
    UInt8EdgeMap isPureLifted_;
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
inline RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
RankFixationClusterPolicy(
    const GraphType & graph,
    const MERGE_PRIOS & mergePrios,
    const NOT_MERGE_PRIOS & notMergePrios,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    mergePriosHist_(graph),
    notMergePriosHist_(graph),
    isLocalEdge_(graph),
    isPureLocal_(graph),
    isPureLifted_(graph),
    edgeSizes_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{

    // minmax
    double minMergePrio = std::numeric_limits<double>::infinity();
    double maxMergePrio = -std::numeric_limits<double>::infinity();
    double minNotMergePrio = std::numeric_limits<double>::infinity();
    double maxNotMergePrio = -std::numeric_limits<double>::infinity();

    graph_.forEachEdge([&](const uint64_t edge){



        minMergePrio = std::min(minMergePrio, double(mergePrios[edge]));
        maxMergePrio = std::max(maxMergePrio, double(mergePrios[edge]));
        minNotMergePrio = std::min(minNotMergePrio, double(notMergePrios[edge]));
        maxNotMergePrio = std::max(maxNotMergePrio, double(notMergePrios[edge]));

    });


    graph_.forEachEdge([&](const uint64_t edge){

        mergePriosHist_[edge].assign(minMergePrio,maxMergePrio,settings_.numberOfBins);
        notMergePriosHist_[edge].assign(minNotMergePrio,maxNotMergePrio,settings_.numberOfBins);

        isLocalEdge_[edge] = isLocalEdge[edge];
        edgeSizes_[edge] = edgeSizes[edge];

        if(settings_.zeroInit){
            if(isLocalEdge_[edge]){
                mergePriosHist_[edge].insert(mergePrios[edge], edgeSizes_[edge]);
                notMergePriosHist_[edge].insert(0.0, edgeSizes_[edge]);
            }
            else{
                mergePriosHist_[edge].insert(0, edgeSizes_[edge]);
                notMergePriosHist_[edge].insert(notMergePrios[edge], edgeSizes_[edge]);
            }
        }
        else{
            mergePriosHist_[edge].insert(mergePrios[edge], edgeSizes_[edge]);
            notMergePriosHist_[edge].insert(notMergePrios[edge], edgeSizes_[edge]);
        }
        
        isPureLocal_[edge] = isLocalEdge[edge];
        isPureLifted_[edge] = !isLocalEdge[edge];

       
        pq_.push(edge, this->pqMergePrio(edge));
    });
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {    
    return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
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
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
pqMergePrio(
    const uint64_t edge
) const {
    return isLocalEdge_[edge] ?  this->getMergePrio(edge) : -1.0; 
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    //std::cout<<"contract edge: "<<edgeToContract<<"\n"; 
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){

}

template<class GRAPH, bool ENABLE_UCM>
inline void 
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    //std::cout<<"    merge edges: a/d "<<aliveEdge<<" "<<deadEdge<<" \n"; 
    NIFTY_CHECK_OP(aliveEdge,!=,deadEdge,"");
    NIFTY_CHECK(pq_.contains(aliveEdge),"");
    NIFTY_CHECK(pq_.contains(deadEdge),"");
    

    pq_.deleteItem(deadEdge);

   


    //  sizes
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];

    const auto zi = settings_.zeroInit ;


    // update merge prio
    if(zi && isPureLifted_[aliveEdge] && !isPureLifted_[deadEdge]){
        mergePriosHist_[aliveEdge] = mergePriosHist_[deadEdge];
    }
    else if(zi && !isPureLifted_[aliveEdge] && isPureLifted_[deadEdge]){
        mergePriosHist_[deadEdge] = mergePriosHist_[aliveEdge];
    }
    else{
        mergePriosHist_[aliveEdge].merge(mergePriosHist_[deadEdge]);
    }


    // update notMergePrio
    if(zi && isPureLocal_[aliveEdge] && !isPureLocal_[deadEdge]){
        notMergePriosHist_[aliveEdge] = notMergePriosHist_[deadEdge];
    }
    else if(zi && !isPureLocal_[aliveEdge] && isPureLocal_[deadEdge]){
        notMergePriosHist_[aliveEdge] = notMergePriosHist_[deadEdge];
    }
    else{
        notMergePriosHist_[aliveEdge].merge(notMergePriosHist_[deadEdge]);
    }

   
    

    edgeSizes_[aliveEdge] = sa + sd;

    const auto deadIsLocalEdge = isLocalEdge_[deadEdge];
    auto & aliveIsLocalEdge = isLocalEdge_[aliveEdge];
    aliveIsLocalEdge = deadIsLocalEdge || aliveIsLocalEdge;

    isPureLocal_[aliveEdge] = isPureLocal_[aliveEdge] && isPureLocal_[deadEdge];
    isPureLifted_[aliveEdge] = isPureLifted_[aliveEdge] && isPureLifted_[deadEdge];


    
    pq_.push(aliveEdge, this->pqMergePrio(aliveEdge));
    
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
RankFixationClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    //std::cout<<"contract edge done: "<<edgeToContract<<"\n\n";
}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

