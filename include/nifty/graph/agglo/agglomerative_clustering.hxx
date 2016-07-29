#pragma once
#ifndef NIFTY_GRAPH_AGGLO_AGGLOMERATIVE_CLUSTERING_HXX
#define NIFTY_GRAPH_AGGLO_AGGLOMERATIVE_CLUSTERING_HXX

#include "nifty/graph/subgraph_mask.hxx"

namespace nifty{
namespace graph{
namespace agglo{



template<class CLUSTER_POLICY>
class AgglomerativeClustering{
public:
    typedef CLUSTER_POLICY ClusterPolicyType;

    AgglomerativeClustering(ClusterPolicyType & clusterPolicy)
    :  clusterPolicy_(clusterPolicy){

    }
private:
    ClusterPolicyType & clusterPolicy_;
};


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_AGGLO_AGGLOMERATIVE_CLUSTERING_HXX*/
