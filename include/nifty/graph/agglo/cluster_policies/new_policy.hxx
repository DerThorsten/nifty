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
class NewPolicy{

    typedef NewPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

private:    
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
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

        enum UpdateRule{
            SMOOTH_MAX,
            GENERALIZED_MEAN
            //HISTOGRAM_RANK
        };

        UpdateRule updateRule0{GENERALIZED_MEAN};
        UpdateRule updateRule1{GENERALIZED_MEAN};

        bool zeroInit = false;
        double p0{1.0};
        double p1{1.0};
        double gamma{0.9}
        uint64_t numberOfNodesStop{1};
        //uint64_t numberOfBins{40};
    };


    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types


    typedef nifty::tools::ChangeablePriorityQueue< double , std::greater<double> > QueueType;

public:

    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE, class EDGE_SIZES>
    NewPolicy(const GraphType &, 
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

    const double getMergePrio(const uint64_t edge)const{
        return mergePrios_[edge];
    }

    const double notMergePrio(const uint64_t edge)const{
        return notMergePrios_[edge];
    }

    // INPUT
    const GraphType &   graph_;



    EdgePrioType mergePrios_;
    EdgePrioType notMergePrios_; 

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
inline NewPolicy<GRAPH, ENABLE_UCM>::
NewPolicy(
    const GraphType & graph,
    const MERGE_PRIOS & mergePrios,
    const NOT_MERGE_PRIOS & notMergePrios,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    mergePrios_(graph),
    notMergePrios_(graph),
    isLocalEdge_(graph),
    isPureLocal_(graph),
    isPureLifted_(graph),
    edgeSizes_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
   
    graph_.forEachEdge([&](const uint64_t edge){
        isLocalEdge_[edge] = isLocalEdge[edge];

        notMergePrios_[edge] = notMergePrios[edge];
        mergePrios_[edge] = mergePrios[edge];

        if(settings_.zeroInit){
            if(isLocalEdge_[edge]) 
                notMergePrios_[edge] = 0.0;
            else
                mergePrios_[edge] = 0.0;
        }
        
        isPureLocal_[edge] = isLocalEdge[edge];
        isPureLifted_[edge] = !isLocalEdge[edge];

        edgeSizes_[edge] = edgeSizes[edge];
        pq_.push(edge, this->pqMergePrio(edge));
    });
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
NewPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {    
    return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
NewPolicy<GRAPH, ENABLE_UCM>::
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
NewPolicy<GRAPH, ENABLE_UCM>::
pqMergePrio(
    const uint64_t edge
) const {
    return isLocalEdge_[edge] ?  mergePrios_[edge] : -1.0; 
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
NewPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    //std::cout<<"contract edge: "<<edgeToContract<<"\n"; 
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename NewPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
NewPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
NewPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){

}

template<class GRAPH, bool ENABLE_UCM>
inline void 
NewPolicy<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    //std::cout<<"    merge edges: a/d "<<aliveEdge<<" "<<deadEdge<<" \n"; 
    NIFTY_CHECK_OP(aliveEdge,!=,deadEdge,"");
    NIFTY_CHECK(pq_.contains(aliveEdge),"");
    NIFTY_CHECK(pq_.contains(deadEdge),"");
    

    pq_.deleteItem(deadEdge);

   

    auto generalized_mean = [](
        const long double a,
        const long double d,
        const long double wa,
        const long double wd,
        const long double p
    ){
        const long double  eps = 0.000000001;
        if(std::isinf(p)){
            // max
            if(p>0){
                return std::max(a,d);
            }
            // min
            else{
                return std::min(a,d);
            }
        }
        else if(p > 1.0-eps && p< 1.0+ eps){
            return (wa*a + wd*d)/(wa+wd);
        }
        else{
            const auto wad = wa+wd;
            const auto nwa = wa/wad;
            const auto nwd = wd/wad;
            const auto sa = nwa * std::pow(a, p);
            const auto sd = nwd * std::pow(d, p);
            return std::pow(sa+sd, 1.0/p);
        }
    };

    auto smooth_max = [](
        const long double a,
        const long double d,
        const long double wa,
        const long double wd,
        const long double p
    ){
        const long double  eps = 0.000000001;
        if(std::isinf(p)){
            // max
            if(p>0){
                return std::max(a,d);
            }
            // min
            else{
                return std::min(a,d);
            }
        }
        else if(p > 0.0-eps && p< 0.0+ eps){
            return (wa*a + wd*d)/(wa+wd);
        }
        else{

            const auto eaw = wa * std::exp(a*p);
            const auto edw = wd * std::exp(d*p);
            return (a*eaw + d*edw)/(eaw + edw);
        }
    };


    //  sizes
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];
    const auto zi = settings_.zeroInit ;



    // update merge prio
    if(zi && isPureLifted_[aliveEdge] && !isPureLifted_[deadEdge]){
        mergePrios_[aliveEdge] = mergePrios_[deadEdge];
    }
    else if(zi && !isPureLifted_[aliveEdge] && isPureLifted_[deadEdge]){
        mergePrios_[deadEdge] = mergePrios_[aliveEdge];
    }
    else{
        if(settings_.updateRule0 == SettingsType::GENERALIZED_MEAN){
            mergePrios_[aliveEdge]    = generalized_mean(mergePrios_[aliveEdge],     mergePrios_[deadEdge],    sa, sd, settings_.p0);
        }
        else if(settings_.updateRule0 == SettingsType::SMOOTH_MAX){
            mergePrios_[aliveEdge]    = smooth_max(mergePrios_[aliveEdge],     mergePrios_[deadEdge],    sa, sd, settings_.p0);
        }
        else{
            NIFTY_CHECK(false,"not yet implemented");
        }
    }


    // update notMergePrio
    if(zi && isPureLocal_[aliveEdge] && !isPureLocal_[deadEdge]){
        notMergePrios_[aliveEdge] = notMergePrios_[deadEdge];
    }
    else if(zi && !isPureLocal_[aliveEdge] && isPureLocal_[deadEdge]){
        notMergePrios_[aliveEdge] = notMergePrios_[deadEdge];
    }
    else{
        if(settings_.updateRule0 == SettingsType::GENERALIZED_MEAN){
            notMergePrios_[aliveEdge] = generalized_mean(notMergePrios_[aliveEdge] , notMergePrios_[deadEdge], sa, sd, settings_.p1);
        }
        else if(settings_.updateRule0 == SettingsType::SMOOTH_MAX){
            notMergePrios_[aliveEdge] = smooth_max(notMergePrios_[aliveEdge] , notMergePrios_[deadEdge], sa, sd, settings_.p1);
        }
        else{
            NIFTY_CHECK(false,"not yet implemented");
        }
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
NewPolicy<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    //std::cout<<"contract edge done: "<<edgeToContract<<"\n\n";
}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

