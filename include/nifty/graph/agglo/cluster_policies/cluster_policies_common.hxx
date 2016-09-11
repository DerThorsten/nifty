#pragma once
#ifndef NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_CLUSTER_POLICIES_COMMON_HXX
#define NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_CLUSTER_POLICIES_COMMON_HXX


namespace nifty{
namespace graph{
namespace agglo{



struct EdgeWeightedClusterPolicySettings{
    double sizeRegularizer{0.5};
    uint64_t numberOfNodesStop{1};
    uint64_t numberOfEdgesStop{0};
};



} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_AGGLO_CLUSTER_POLICIES_CLUSTER_POLICIES_COMMON_HXX*/
