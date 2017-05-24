#pragma once


#include <functional>



#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"

namespace nifty{
namespace graph{
namespace agglo{

template<class GRAPH>
class MinimumNodeSizeClusterPolicy{

    typedef MinimumNodeSizeClusterPolicy<GRAPH> SelfType;

private:

    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;

public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgeIndicatorsType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;
    struct Settings{
        double minimumNodeSize{1.0};
        double sizeRegularizer{0.5};
        double gamma{0.999};
    };
    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, false> ;
private:

    // internal types


    typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES>
    MinimumNodeSizeClusterPolicy(const GraphType &, 
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const NODE_SIZES & ,
                              const Settings & settings = Settings());


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
    NodeSizesType       nodeSizes_;
    Settings            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

};


template<class GRAPH>
template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES>
inline MinimumNodeSizeClusterPolicy<GRAPH>::
MinimumNodeSizeClusterPolicy(
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
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
    graph_.forEachEdge([&](const uint64_t edge){
        edgeIndicators_[edge] = edgeIndicators[edge];
        edgeSizes_[edge] = edgeSizes[edge];
    });
    graph_.forEachNode([&](const uint64_t node){
        nodeSizes_[node] = nodeSizes[node];
    });
    this->initializeWeights();
}

template<class GRAPH>
inline std::pair<uint64_t, double> 
MinimumNodeSizeClusterPolicy<GRAPH>::
edgeToContractNext() const {
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH>
inline bool 
MinimumNodeSizeClusterPolicy<GRAPH>::
isDone() const {

    const auto topEdge = pq_.top();
    const auto uv = edgeContractionGraph_.uv(topEdge);
    const auto sizeU = nodeSizes_[uv.first];
    const auto sizeV = nodeSizes_[uv.second];


    //std::cout<<"top "<<pq_.topPriority()<<" "<<sizeU<<" "<<sizeV<<"\n";

    if(edgeContractionGraph_.numberOfNodes() <= 1 || edgeContractionGraph_.numberOfEdges() <= 0){
        return  true;
    }
    // edges where both nodes are large as the min.
    // size limit have an edge weight of inf.
    // if this is the nextbest edge to contract we are done
    return std::isinf(pq_.topPriority());
}


template<class GRAPH>
inline void 
MinimumNodeSizeClusterPolicy<GRAPH>::
initializeWeights() {
    for(const auto edge : graph_.edges())
        pq_.push(edge, this->computeWeight(edge));
}

template<class GRAPH>
inline double 
MinimumNodeSizeClusterPolicy<GRAPH>::
computeWeight(
    const uint64_t edge
) const {

    const auto uv = edgeContractionGraph_.uv(edge);
    const auto sizeU = nodeSizes_[uv.first];
    const auto sizeV = nodeSizes_[uv.second];

    // basic edge weight weight (with size regularizer)
    const auto sr = settings_.sizeRegularizer;
    const auto sFac = 2.0 / ( 1.0/std::pow(sizeU,sr) + 1.0/std::pow(sizeV,sr) );
    const auto w =  edgeIndicators_[edge] * sFac;

    const auto minSize =settings_.minimumNodeSize;

    if( sizeU < minSize && sizeV < minSize){
        return w ;
    }
    else if(sizeU < minSize || sizeV < minSize){
        return w * settings_.gamma;
    }
    else{
        return std::numeric_limits<double>::infinity();
    }
}


template<class GRAPH>
inline void 
MinimumNodeSizeClusterPolicy<GRAPH>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH>
inline typename MinimumNodeSizeClusterPolicy<GRAPH>::EdgeContractionGraphType & 
MinimumNodeSizeClusterPolicy<GRAPH>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH>
inline void 
MinimumNodeSizeClusterPolicy<GRAPH>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    nodeSizes_[aliveNode] +=nodeSizes_[deadNode];
}

template<class GRAPH>
inline void 
MinimumNodeSizeClusterPolicy<GRAPH>::
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

template<class GRAPH>
inline void 
MinimumNodeSizeClusterPolicy<GRAPH>::
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
