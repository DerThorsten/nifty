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
    typedef typename ClusterPolicyType::EdgeContractionGraphType::WithEdgeUfd WithEdgeUfd;
    typedef typename ClusterPolicyType::GraphType GraphType;
    typedef typename ClusterPolicyType::EdgeContractionGraphType EdgeContractionGraphType;
    AgglomerativeClustering(ClusterPolicyType & clusterPolicy)
    :  clusterPolicy_(clusterPolicy){

    }

    void run(){
        while(!clusterPolicy_.isDone()){

            if(clusterPolicy_.edgeContractionGraph().numberOfEdges() == 0)
                break;

            const auto edgeToContractNextAndPriority = clusterPolicy_.edgeToContractNext();
            const auto edgeToContractNext = edgeToContractNextAndPriority.first;
            const auto priority = edgeToContractNextAndPriority.second;
            clusterPolicy_.edgeContractionGraph().contractEdge(edgeToContractNext);
        }
    }
    

    template<class EDGE_DENDROGRAM_HEIGHT>
    void runAndGetDendrogramHeight(EDGE_DENDROGRAM_HEIGHT & dendrogramHeight){


        while(!clusterPolicy_.isDone()){
            const auto edgeToContractNextAndPriority = clusterPolicy_.edgeToContractNext();
            const auto edgeToContractNext = edgeToContractNextAndPriority.first;
            const auto priority = edgeToContractNextAndPriority.second;
            dendrogramHeight[edgeToContractNext] = priority;
            clusterPolicy_.edgeContractionGraph().contractEdge(edgeToContractNext);
        }

        this->ucmTransform(dendrogramHeight);
    }

    template<class EDGE_MAP>
    void ucmTransform(EDGE_MAP & edgeMap)const{

        const auto & cgraph = clusterPolicy_.edgeContractionGraph();
        this->graph().forEachEdge([&](const uint64_t edge){
            const auto reprEdge = cgraph.findRepresentativeEdge(edge);
            edgeMap[edge] = edgeMap[reprEdge];
        });

    }


    template<class EDGE_MAP, class EDGE_MAP_OUT>
    void ucmTransform(const EDGE_MAP & edgeMap, EDGE_MAP_OUT & edgeMapOut)const{

        const auto & cgraph = clusterPolicy_.edgeContractionGraph();
        this->graph().forEachEdge([&](const uint64_t edge){
            const auto reprEdge = cgraph.findRepresentativeEdge(edge);
            edgeMapOut[edge] = edgeMap[reprEdge];
        });

    }

    

    const GraphType & graph()const{
        return clusterPolicy_.edgeContractionGraph().graph();
    }

    template<class NODE_MAP>
    void result(NODE_MAP & nodeMap)const{
        const auto & cgraph = clusterPolicy_.edgeContractionGraph();
        const auto & graph = cgraph.graph();
        for(const auto node : graph.nodes()){
            nodeMap[node] = cgraph.findRepresentativeNode(node);
        }
    }
private:
    ClusterPolicyType & clusterPolicy_;
};


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_AGGLO_AGGLOMERATIVE_CLUSTERING_HXX*/
