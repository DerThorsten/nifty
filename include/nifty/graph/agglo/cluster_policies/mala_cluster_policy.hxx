#pragma once

#include <functional>
#include <array>


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
class MalaClusterPolicy{

    typedef MalaClusterPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

private:
    typedef typename GRAPH:: template EdgeMap<uint64_t> UInt64EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;

public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgeIndicatorsType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;


    typedef UInt64EdgeMap                               MergeTimesType;

    struct SettingsType : public EdgeWeightedClusterPolicySettings
    {

        float threshold{0.5};
        bool verbose{false};
        int bincount{40};
    };
    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types
    const static std::size_t NumberOfBins = 20;
    typedef nifty::histogram::Histogram<float> HistogramType;
    //typedef std::array<float, NumberOfBins> HistogramType;     
    typedef typename GRAPH:: template EdgeMap<HistogramType> EdgeHistogramMap;


    typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES>
    MalaClusterPolicy(const GraphType &, 
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const NODE_SIZES & ,
                              const SettingsType & settings = SettingsType());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone() const;

    // callback called by edge contraction graph
    
    EdgeContractionGraphType & edgeContractionGraph();


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
    const MergeTimesType & mergeTimes() const {
        return mergeTimes_;
    }
    const NodeSizesType & nodeSizes() const {
        return nodeSizes_;
    }

private:
    float histogramToMedian(const uint64_t edge) const;

    // const EdgeIndicatorsType & edgeIndicators() const {
    //     return edgeIndicators_;
    // }
    // const EdgeSizesType & edgeSizes() const {
    //     return edgeSizes_;
    // }
    // const NodeSizesType & nodeSizes() const {
    //     return nodeSizes_;
    // }
    
private:
    // INPUT
    const GraphType &   graph_;
    EdgeIndicatorsType  edgeIndicators_;
    EdgeSizesType       edgeSizes_;
    NodeSizesType       nodeSizes_;


    MergeTimesType       mergeTimes_;


    SettingsType            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

    EdgeHistogramMap histograms_;

    uint64_t time_;


};


template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES>
inline MalaClusterPolicy<GRAPH, ENABLE_UCM>::
MalaClusterPolicy(
    const GraphType & graph,
    const EDGE_INDICATORS & edgeIndicators,
    const EDGE_SIZES      & edgeSizes,
    const NODE_SIZES      & nodeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    edgeIndicators_(graph),
    edgeSizes_(graph),
    nodeSizes_(graph),
    mergeTimes_(graph, graph_.numberOfNodes()),
    settings_(settings),
    edgeContractionGraph_(graph, *this),
    pq_(graph.edgeIdUpperBound()+1),
    histograms_(graph, HistogramType(0,1,settings.bincount)),
    time_(0)
{
    graph_.forEachEdge([&](const uint64_t edge){


        const auto val = edgeIndicators[edge];


        // currently the value itself
        // is the median
        const auto size = edgeSizes[edge];
        edgeSizes_[edge] = size;

        // put in histogram
        histograms_[edge].insert(val, size);

        // put in pq
        pq_.push(edge, val);

    });

    graph_.forEachNode([&](const uint64_t node){
        nodeSizes_[node] = nodeSizes[node];
    });
    //this->initializeWeights();
}

template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
isDone() const {
    if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop)
        return  true;
    if(edgeContractionGraph_.numberOfEdges() <= settings_.numberOfEdgesStop)
        return  true;
    if(pq_.topPriority() >= settings_.threshold)
        return  true;
    return false;
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    mergeTimes_[edgeToContract] = time_;
    ++time_;
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, bool ENABLE_UCM>
inline typename MalaClusterPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType & 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    pq_.deleteItem(deadEdge);

    // merging the histogram is just adding
    auto & ha = histograms_[aliveEdge];
    auto & hd = histograms_[deadEdge];
    ha.merge(hd);
    pq_.push(aliveEdge, histogramToMedian(aliveEdge));
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){

}

template<class GRAPH, bool ENABLE_UCM>
inline float 
MalaClusterPolicy<GRAPH, ENABLE_UCM>::
histogramToMedian(
    const uint64_t edge
) const{
    // todo optimize me
    float median;
    const float rank=0.5;
    nifty::histogram::quantiles(histograms_[edge],&rank,&rank+1,&median);
    return median;
}





} // namespace agglo
} // namespace nifty::graph
} // namespace nifty


