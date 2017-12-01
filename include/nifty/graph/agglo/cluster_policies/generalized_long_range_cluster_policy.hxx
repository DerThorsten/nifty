#pragma once

#include <functional>


#include "nifty/histogram/histogram.hxx"
#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"

namespace nifty{
namespace graph{
namespace agglo{





template<
    class GRAPH,bool ENABLE_UCM
>
class GeneralizedLongRangeClusterPolicy{

    typedef GeneralizedLongRangeClusterPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

private:
    typedef typename GRAPH:: template EdgeMap<uint8_t>  UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double>   FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double>   FloatNodeMap;
    typedef typename GRAPH:: template NodeMap<uint64_t> UInt64NodeMap;
public:

    // input types
    typedef GRAPH                                        GraphType;
    typedef FloatEdgeMap                                 EdgeIndicatorsType;
    typedef FloatEdgeMap                                 EdgeSizesType;
    typedef UInt8EdgeMap                                 EdgeIsLocalType;
    typedef FloatNodeMap                                 NodeSizesType;
    typedef UInt64NodeMap                                SeedsType;

    struct SettingsType{
        double   stopPriority{std::numeric_limits<float>::infinity()};
        uint64_t stopNodeNumber{1};
        bool     useSeeds{false};
        double   minSize{0.0};
        double   sizeRegularizer{0.0};
    };
    
    typedef EdgeContractionGraph<GraphType, SelfType>    EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types
    typedef nifty::histogram::Histogram<float> HistogramType;
    typedef typename GRAPH:: template EdgeMap<HistogramType> HistogramMap;

    typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    template<
        class EDGE_INDICATORS, 
        class EDGE_SIZES,
        class NODE_SIZES,
        class IS_LOCAL_EDGE,
        class SEEDS
    >
    GeneralizedLongRangeClusterPolicy(const GraphType &, 
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const NODE_SIZES & ,
                              const IS_LOCAL_EDGE &,
                              const SEEDS & ,
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
    EdgeIsLocalType     isLocalEdge_;
    NodeSizesType       nodeSizes_;
    SeedsType           seeds_;
    SettingsType        settings_;
    
    // INTERNAL
    //HistogramMap eHist_;
     
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

};


template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS, class EDGE_SIZES,class NODE_SIZES, class IS_LOCAL_EDGE, class SEEDS>
inline GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
GeneralizedLongRangeClusterPolicy(
    const GraphType & graph,
    const EDGE_INDICATORS & edgeIndicators,
    const EDGE_SIZES      & edgeSizes,
    const NODE_SIZES      & nodeSizes,
    const IS_LOCAL_EDGE & isLocalEdge,
    const SEEDS &         seeds,
    const SettingsType & settings
)
:   graph_(graph),
    edgeIndicators_(graph),
    edgeSizes_(graph),
    isLocalEdge_(graph),
    nodeSizes_(graph),
    seeds_(graph),
    settings_(settings),
    //eHist_(graph, HistogramType(0.0, 1.0, 40)),
    edgeContractionGraph_(graph, *this),
    pq_(graph.edgeIdUpperBound()+1)
{
    graph_.forEachEdge([&](const uint64_t edge){

        edgeIndicators_[edge] = edgeIndicators[edge];
        //eHist_[edge].insert(edgeIndicators[edge]);

        edgeSizes_[edge] = edgeSizes[edge];
        isLocalEdge_[edge] = isLocalEdge[edge];
    });
    graph_.forEachNode([&](const uint64_t node){
        nodeSizes_[node] = nodeSizes[node];
        seeds_[node] = seeds[node];
    });
    this->initializeWeights();
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {
    const auto edgeToContract = pq_.top();
    NIFTY_CHECK(isLocalEdge_[edgeToContract], "internal error");
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
isDone() const {

    
    const auto topPriority =  pq_.topPriority();
    if(topPriority >= settings_.stopPriority){
        return true;
    }
    else if(edgeContractionGraph_.numberOfNodes() <= settings_.stopNodeNumber){
        return true;
    }
    else if(edgeContractionGraph_.numberOfEdges() == 0){
        return true;
    }
    else if(pq_.empty()){
       return true;
    }
    else{
        return false;
    }
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
initializeWeights() {
    for(const auto edge : graph_.edges())
        pq_.push(edge, this->computeWeight(edge));
}

template<class GRAPH, bool ENABLE_UCM>
inline double 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
computeWeight(
    const uint64_t edge
) const {

    const auto uv = edgeContractionGraph_.uv(edge);
    double srFac = 1.0;
    if(settings_.sizeRegularizer > 0.0000001){

        const auto uv = edgeContractionGraph_.uv(edge);
        const auto sizeU = nodeSizes_[uv.first];
        const auto sizeV = nodeSizes_[uv.second];
        const auto sr = settings_.sizeRegularizer;
        srFac = 2.0 / (1.0/std::pow(sizeU,sr)+ 1.0/std::pow(sizeV,sr));
    }


    if(!isLocalEdge_[edge]){
        return std::numeric_limits<float>::infinity();
    }

    if(settings_.useSeeds){
        const auto seedU = seeds_[uv.first];
        const auto seedV = seeds_[uv.second];

        if(seedU==0 && seedV==0){
            //return srFac*edgeIndicators_[edge];
            return std::numeric_limits<float>::infinity();
        }
        else if(seedU == seedV){
            return 0.0;
        }
        else if(seedU!=0 && seedV!=0){
            return std::numeric_limits<float>::infinity();
        }
        else{
            return srFac*edgeIndicators_[edge];
        }
    }
    else{
        return srFac*edgeIndicators_[edge];
    }

    
    
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    nodeSizes_[aliveNode] +=nodeSizes_[deadNode];
    if(settings_.useSeeds){
        seeds_[aliveNode] = std::max(seeds_[aliveNode],  seeds_[deadNode]);
    }
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    pq_.deleteItem(deadEdge);
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];
    const auto s = sa + sd;


    auto & la = isLocalEdge_[aliveEdge];
    const auto & ld = isLocalEdge_[deadEdge];

    la = la || ld;

    //eHist_[aliveEdge].merge(eHist_[deadEdge]);

    edgeIndicators_[aliveEdge] = (sa*edgeIndicators_[aliveEdge] + sd*edgeIndicators_[deadEdge])/s;
    edgeSizes_[aliveEdge] = s;

    //pq_.push(aliveEdge, computeWeight(aliveEdge));
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
GeneralizedLongRangeClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    //// HERE WE UPDATE 
    const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
    for(auto adj : edgeContractionGraph_.adjacency(u)){
        const auto edge = adj.edge();
        pq_.push(edge, computeWeight(edge));
    }
}




} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

