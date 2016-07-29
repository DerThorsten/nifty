#pragma once
#ifndef NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX
#define NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX

#include <functional>

#include "vigra/priority_queue.hxx"

namespace nifty{
namespace graph{
namespace agglo{



struct EdgeWeightedClusterPolicySettings{

};

template<
    class GRAPH,
    class EDGE_INDICATORS,
    class EDGE_SIZES,
    class NODE_SIZES
>
class EdgeWeightedClusterPolicy{

public:
    // input types
    typedef GRAPH                               GraphType;
    typedef EDGE_INDICATORS                     EdgeIndicatorsType;
    typedef EDGE_SIZES                          EdgeSizesType;
    typedef NODE_SIZES                          NodeSizesType;
    typedef EdgeWeightedClusterPolicySettings   Settings;
private:

    // internal types
    typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

    EdgeWeightedClusterPolicy(
        const GraphType & graph,
        EdgeIndicatorsType  edgeIndicators,
        EdgeSizesType       edgeSizes,
        NodeSizesType       nodeSizes,
        const Settings & settings = Settings()
    )
    :
    graph_(graph),
    edgeIndicators_(edgeIndicators),
    edgeSizes_(edgeSizes),
    nodeSizes_(nodeSizes),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings)
    {

    }

private:
    

    
    // INPUT
    const GraphType &   graph_;
    EdgeIndicatorsType  edgeIndicators_;
    EdgeSizesType       edgeSizes_;
    NodeSizesType       nodeSizes_;
    Settings            settings_;
    // INTERNAL
    QueueType pq_;

};



} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_EDGE_WEIGHTED_CLUSTER_POLICY_HXX*/
