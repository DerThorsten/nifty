#pragma once

#include <iostream>
#include "nifty/graph/subgraph_mask.hxx"

namespace nifty{
namespace graph{
namespace agglo{




// forward declarations
template<class AGGLOMERATIVE_CLUSTERING>
class AgglomerativeClustering;



template<class AGGLOMERATIVE_CLUSTERING>
class DendrogramAgglomerativeClusteringVisitor{
private:

    typedef std::tuple<uint64_t,uint64_t, double, double> MergeEncodingType; 

public:
    typedef AGGLOMERATIVE_CLUSTERING AgglomerativeClusteringType;
    typedef typename AgglomerativeClusteringType::GraphType GraphType;

    typedef typename GraphType:: template NodeMap<double> NodeToEncoding;
    typedef typename GraphType:: template NodeMap<double> NodeSize;


    DendrogramAgglomerativeClusteringVisitor(
        const AgglomerativeClusteringType & agglomerativeClustering
    )
    :   nodeSizes_(agglomerativeClustering.graph()),
        agglomerativeClustering_(agglomerativeClustering),
        nodeToEncoding_(agglomerativeClustering.graph()),
        encoding_(),
        timeStamp_(agglomerativeClustering.graph().numberOfNodes())
    {
        for( auto node : agglomerativeClustering.graph().nodes()){
            nodeToEncoding_[node] = node;
            nodeSizes_[node] = 1.0;
        }
    }

    template<class NODE_SIZES>
    DendrogramAgglomerativeClusteringVisitor(
        const AgglomerativeClusteringType & agglomerativeClustering,
        NODE_SIZES & nodeSizes
    )
    :   nodeSizes_(agglomerativeClustering.graph()),
        agglomerativeClustering_(agglomerativeClustering),
        nodeToEncoding_(agglomerativeClustering.graph()),
        encoding_(),
        timeStamp_(agglomerativeClustering.graph().numberOfNodes())
    {
        for( auto node : agglomerativeClustering.graph().nodes()){
            nodeToEncoding_[node] = node;
            nodeSizes_[node] = nodeSizes[node];
        }
    }


    bool isDone()const{
        return false;
    }

    void visit(const uint64_t aliveNode, const uint64_t deadNode, const double p){
        //std::cout<<"a "<<aliveNode<<" d "<<deadNode<<" p "<<p<<"\n";

        auto ea = nodeToEncoding_[aliveNode];
        auto ed = nodeToEncoding_[deadNode];

        nodeSizes_[aliveNode] += nodeSizes_[deadNode];

        nodeToEncoding_[aliveNode] = timeStamp_;
        nodeToEncoding_[deadNode] = timeStamp_;

        encoding_.emplace_back(ea, ed, p, nodeSizes_[aliveNode]);
        ++timeStamp_;

    }
    const auto & agglomerativeClustering()const{
        return agglomerativeClustering_;
    }
    const auto & dendrogramEncoding()const{
        return encoding_;
    }
private:
    const AgglomerativeClusteringType & agglomerativeClustering_;

    NodeSize nodeSizes_;
    NodeToEncoding nodeToEncoding_;
    std::vector<MergeEncodingType > encoding_;
    uint64_t timeStamp_;
};

// template<class AGGLOMERATIVE_CLUSTERING>
// class EmptyAgglomerativeClusteringVisitor{
// public:
//     typedef AGGLOMERATIVE_CLUSTERING AgglomerativeClusteringType;

// private:
    
// };




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

    template<class VISITOR>
    void run(
        VISITOR & visitor, const bool verbose=false, const uint64_t printNth=100
    ){

        const auto & cgraph = clusterPolicy_.edgeContractionGraph();

        while(!clusterPolicy_.isDone()  && !visitor.isDone() ){


            if(clusterPolicy_.edgeContractionGraph().numberOfEdges() == 0)
                break;
            //std::cout<<"AgglomerativeClustering edgeToContractNext\n";
            const auto edgeToContractNextAndPriority = clusterPolicy_.edgeToContractNext();
            //std::cout<<"AgglomerativeClustering edgeToContractNext done\n";
            const auto edgeToContractNext = edgeToContractNextAndPriority.first;
            const auto priority = edgeToContractNextAndPriority.second;


            if(verbose){
                
                const auto nNodes = cgraph.numberOfNodes();
                if(  (nNodes + 1) % printNth  == 0){
                    std::cout<<"Nodes "<<cgraph.numberOfNodes()<<" p="<<priority<<"\n";
                }
            }


            const auto uv = cgraph.uv(edgeToContractNext);
            clusterPolicy_.edgeContractionGraph().contractEdge(edgeToContractNext);

            const auto aliveNode =cgraph.findRepresentativeNode(uv.first);
            const auto deadNode = uv.first == aliveNode ? uv.second : uv.first;

            visitor.visit(aliveNode, deadNode, priority);
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


    // for each edge, this returns the edge id which was the most recent merge for this
    // node. This CANNOT be used for a merge hierarchy
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


    // for each n
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

    const ClusterPolicyType & clusterPolicy()const{
        return clusterPolicy_;
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

