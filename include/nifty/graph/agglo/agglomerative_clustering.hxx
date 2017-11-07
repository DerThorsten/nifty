#pragma once

#include <iostream>
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


    void run(const bool verbose=false, const uint64_t printNth=100){
        while(!clusterPolicy_.isDone()){

            if(clusterPolicy_.edgeContractionGraph().numberOfEdges() == 0)
                break;
            //std::cout<<"AgglomerativeClustering edgeToContractNext\n";
            const auto edgeToContractNextAndPriority = clusterPolicy_.edgeToContractNext();
            //std::cout<<"AgglomerativeClustering edgeToContractNext done\n";
            const auto edgeToContractNext = edgeToContractNextAndPriority.first;
            const auto priority = edgeToContractNextAndPriority.second;
            if(verbose){
                const auto & cgraph = clusterPolicy_.edgeContractionGraph();
                const auto nNodes = cgraph.numberOfNodes();
                if(  (nNodes + 1) % printNth  == 0){
                    std::cout<<"Nodes "<<cgraph.numberOfNodes()<<" p="<<priority<<"\n";
                }
            }
            clusterPolicy_.edgeContractionGraph().contractEdge(edgeToContractNext);
        }
    }

  

    template<class MERGE_TIMES, class EDGE_DENDROGRAM_HEIGHT>
    void runAndGetMergeTimesAndDendrogramHeight(
        MERGE_TIMES & mergeTimes,
        EDGE_DENDROGRAM_HEIGHT & dendrogramHeight,
        const bool verbose=false  
    ){
        const auto & cgraph = clusterPolicy_.edgeContractionGraph();
        const auto & graph = cgraph.graph();
        for(const auto edge: graph.edges()){
            mergeTimes[edge] = graph.numberOfNodes();
        }
        auto t=-0;
        while(!clusterPolicy_.isDone()){

            if(clusterPolicy_.edgeContractionGraph().numberOfEdges() == 0)
                break;

            const auto edgeToContractNextAndPriority = clusterPolicy_.edgeToContractNext();
            const auto edgeToContractNext = edgeToContractNextAndPriority.first;
            const auto priority = edgeToContractNextAndPriority.second;
            dendrogramHeight[edgeToContractNext] = priority;
            mergeTimes[edgeToContractNext] = edgeToContractNext;
            if(verbose){
                const auto & cgraph = clusterPolicy_.edgeContractionGraph();
                std::cout<<"Nodes "<<cgraph.numberOfNodes()<<" p="<<priority<<"\n";
            }
            clusterPolicy_.edgeContractionGraph().contractEdge(edgeToContractNext);
        }
        this->ucmTransform(mergeTimes);
        this->ucmTransform(dendrogramHeight);
    }
    

    template<class MERGE_TIMES>
    void runAndGetMergeTimes(
        MERGE_TIMES & mergeTimes,
        const bool verbose=false  
    ){
        const auto & cgraph = clusterPolicy_.edgeContractionGraph();
        const auto & graph = cgraph.graph();
        for(const auto edge: graph.edges()){
            mergeTimes[edge] = graph.numberOfNodes();
        }
        auto t=-0;
        while(!clusterPolicy_.isDone()){

            if(clusterPolicy_.edgeContractionGraph().numberOfEdges() == 0)
                break;

            const auto edgeToContractNextAndPriority = clusterPolicy_.edgeToContractNext();
            const auto edgeToContractNext = edgeToContractNextAndPriority.first;
            const auto priority = edgeToContractNextAndPriority.second;
            mergeTimes[edgeToContractNext] = edgeToContractNext;
            if(verbose){
                const auto & cgraph = clusterPolicy_.edgeContractionGraph();
                std::cout<<"Nodes "<<cgraph.numberOfNodes()<<" p="<<priority<<"\n";
            }
            clusterPolicy_.edgeContractionGraph().contractEdge(edgeToContractNext);
        }
        this->ucmTransform(mergeTimes);
    }
    

    template<class EDGE_DENDROGRAM_HEIGHT>
    void runAndGetDendrogramHeight(EDGE_DENDROGRAM_HEIGHT & dendrogramHeight,const bool verbose=false){


        while(!clusterPolicy_.isDone()){
            const auto edgeToContractNextAndPriority = clusterPolicy_.edgeToContractNext();
            const auto edgeToContractNext = edgeToContractNextAndPriority.first;
            const auto priority = edgeToContractNextAndPriority.second;
            dendrogramHeight[edgeToContractNext] = priority;
            if(verbose){
                const auto & cgraph = clusterPolicy_.edgeContractionGraph();
                std::cout<<"Nodes "<<cgraph.numberOfNodes()<<" p="<<priority<<"\n";
            }
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

