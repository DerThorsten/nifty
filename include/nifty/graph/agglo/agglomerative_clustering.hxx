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
    typedef typename ClusterPolicyType::GraphType GraphType;
    typedef typename ClusterPolicyType::EdgeContractionGraphType EdgeContractionGraphType;
    AgglomerativeClustering(ClusterPolicyType & clusterPolicy)
    :  clusterPolicy_(clusterPolicy){

    }

    void run(){
        while(!clusterPolicy_.isDone()){
            const auto edgeToContractNext = clusterPolicy_.edgeToContractNext();
            clusterPolicy_.edgeContractionGraph().contractEdge(edgeToContractNext);
        }
    }

    const GraphType & graph()const{
        return clusterPolicy_.edgeContractionGraph().graph();
    }

    template<class NODE_MAP>
    void result(NODE_MAP & nodeMap)const{
        const auto & cgraph = clusterPolicy_.edgeContractionGraph();
        const auto & ufd  = cgraph.ufd();
        const auto & graph = cgraph.graph();
        for(const auto node : graph.nodes()){
            nodeMap[node] = ufd.find(node);
        }
    }
private:
    ClusterPolicyType & clusterPolicy_;
};


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_AGGLO_AGGLOMERATIVE_CLUSTERING_HXX*/
