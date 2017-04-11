#pragma once

#include <functional>
#include <array>


#include "vigra/priority_queue.hxx"
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

    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;

public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgeIndicatorsType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;

    struct Settings : public EdgeWeightedClusterPolicySettings
    {

        float threshold{0.5};
        bool verbose{false};
    };
    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types
    const static size_t NumberOfBins = 20;
    typedef std::array<float, NumberOfBins> HistogramType;     
    typedef typename GRAPH:: template EdgeMap<HistogramType> EdgeHistogramMap;


    typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES>
    MalaClusterPolicy(const GraphType &, 
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const NODE_SIZES & ,
                              const Settings & settings = Settings());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone() const;

    // callback called by edge contraction graph
    
    EdgeContractionGraphType & edgeContractionGraph();


    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);

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
    Settings            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

    EdgeHistogramMap histograms_;



};


template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES>
inline MalaClusterPolicy<GRAPH, ENABLE_UCM>::
MalaClusterPolicy(
    const GraphType & graph,
    const EDGE_INDICATORS & edgeIndicators,
    const EDGE_SIZES      & edgeSizes,
    const NODE_SIZES      & nodeSizes,
    const Settings & settings
)
:   graph_(graph),
    edgeIndicators_(graph),
    edgeSizes_(graph),
    nodeSizes_(graph),
    settings_(settings),
    edgeContractionGraph_(graph, *this),
    pq_(graph.edgeIdUpperBound()+1),
    histograms_(graph)
{
    graph_.forEachEdge([&](const uint64_t edge){


        const auto val = edgeIndicators[edge];


        // currently the value itself
        // is the median
        const auto size = edgeSizes[edge];
        edgeSizes_[edge] = size;

        // put in histogram
        auto bin = std::min(size_t(val*float(NumberOfBins)), size_t(NumberOfBins-1));
        auto & hist = histograms_[edge];
        std::fill(hist.begin(), hist.end(), 0.0);
        hist[bin] = size;

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
    edgeSizes_[aliveEdge] += edgeSizes_[deadEdge];
    for(auto bin=0; bin<NumberOfBins; ++bin){
        ha[bin] += hd[bin];
    }
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
    const auto & hist = histograms_[edge];
    auto bin=0;
    auto fbin=0.0f;

    auto halfSize = edgeSizes_[edge] / 2.0;
    auto accSize  = 0.0;
    while(true){
        auto newAccSize = accSize + hist[bin];
        if(bin == NumberOfBins-1){
            return 1.0;
        }
        else if(newAccSize > halfSize){
            // interpolate the median
            // by first interpolating the
            // float bin index 
            const auto dLow  = halfSize - accSize;
            const auto dHigh = newAccSize - halfSize;

            const auto wHigh = dLow / (dLow+dHigh);
            const auto wLow  = 1.0 - wHigh;

            return   (float(bin)  /float(NumberOfBins))*wLow 
                   + (float(bin+1)/float(NumberOfBins))*wHigh;
        }
        accSize = newAccSize;
        ++bin;
    }
    // we should not come here, but 
    // compilers might be sissies about that
    // and warn
    return 1.0;
}





} // namespace agglo
} // namespace nifty::graph
} // namespace nifty


