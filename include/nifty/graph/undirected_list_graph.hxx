#pragma once
#ifndef NIFTY_GRAPH_SIMPLE_GRAPH_HXX
#define NIFTY_GRAPH_SIMPLE_GRAPH_HXX

#include <vector>
#include <map>
#include <boost/version.hpp>

// for strange reason travis does not find the boost flat set
#ifdef WITHIN_TRAVIS
#include <set>
#define __setimpl std::set
#else
#include <boost/container/flat_set.hpp>
#define __setimpl boost::container::flat_set
#endif


#include <boost/iterator/counting_iterator.hpp>

#include "nifty/container/flat_set.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_graph_base.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/graph_tags.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace graph{




namespace detail_graph{
    template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
    struct UndirectedGraphTypeHelper{
        typedef EDGE_INTERANL_TYPE EdgeInternalType;
        typedef NODE_INTERNAL_TYPE NodeInteralType;
        typedef detail_graph::UndirectedAdjacency<int64_t,int64_t,NodeInteralType,EdgeInternalType> NodeAdjacency;
        //typedef std::set<NodeAdjacency > NodeStorage;
        typedef nifty::container::FlatSet <NodeAdjacency> NodeStorage;

        typedef std::pair<NodeInteralType,NodeInteralType> EdgeStorage;
        typedef boost::counting_iterator<int64_t> NodeIter;
        typedef boost::counting_iterator<int64_t> EdgeIter;
        typedef typename NodeStorage::const_iterator AdjacencyIter;
    };



    class SimpleGraphNodeIter : public boost::counting_iterator<int64_t>{
        using boost::counting_iterator<int64_t>::counting_iterator;
        using boost::counting_iterator<int64_t>::operator=;
    };

    class SimpleGraphEdgeIter : public boost::counting_iterator<int64_t>{
        using boost::counting_iterator<int64_t>::counting_iterator;
        using boost::counting_iterator<int64_t>::operator=;
    };
};


template<class EDGE_INTERANL_TYPE = int64_t, 
         class NODE_INTERNAL_TYPE = int64_t>
class UndirectedGraph : public
    UndirectedGraphBase<
        UndirectedGraph<EDGE_INTERANL_TYPE,NODE_INTERNAL_TYPE>,
        detail_graph::SimpleGraphNodeIter,
        detail_graph::SimpleGraphEdgeIter,
        typename detail_graph::UndirectedGraphTypeHelper<EDGE_INTERANL_TYPE,NODE_INTERNAL_TYPE>::AdjacencyIter
    >
{
protected:
    typedef EDGE_INTERANL_TYPE EdgeInternalType;
    typedef NODE_INTERNAL_TYPE NodeInteralType;
    typedef detail_graph::UndirectedAdjacency<int64_t,int64_t,NodeInteralType,EdgeInternalType> NodeAdjacency;
    typedef nifty::container::FlatSet<NodeAdjacency> NodeStorage;
    typedef std::pair<EdgeInternalType,EdgeInternalType> EdgeStorage;
public:
    typedef detail_graph::SimpleGraphNodeIter NodeIter;
    typedef boost::counting_iterator<int64_t> EdgeIter;
    typedef typename NodeStorage::const_iterator AdjacencyIter;


    typedef ContiguousTag EdgeIdTag;
    typedef ContiguousTag NodeIdTag;

    typedef SortedTag EdgeIdOrderTag;
    typedef SortedTag NodeIdOrderTag;


    // constructors
    UndirectedGraph(const uint64_t numberOfNodes = 0, const uint64_t reserveNumberOfEdges = 0);
    void assign(const uint64_t numberOfNodes = 0, const uint64_t reserveNumberOfEdges = 0);
    int64_t insertEdge(const int64_t u, const int64_t v);




    // MUST IMPL INTERFACE
    int64_t u(const int64_t e)const;
    int64_t v(const int64_t e)const;

    int64_t findEdge(const int64_t u, const int64_t v)const;
    int64_t nodeIdUpperBound() const;
    int64_t edgeIdUpperBound() const;
    uint64_t numberOfEdges() const;
    uint64_t numberOfNodes() const;

    NodeIter nodesBegin()const;
    NodeIter nodesEnd()const;
    EdgeIter edgesBegin()const;
    EdgeIter edgesEnd()const;

    AdjacencyIter adjacencyBegin(const int64_t node)const;
    AdjacencyIter adjacencyEnd(const int64_t node)const;
    AdjacencyIter adjacencyOutBegin(const int64_t node)const;

    // optional (with default impl in base)
    std::pair<int64_t,int64_t> uv(const int64_t e)const;


    template<class F>
    void forEachEdge(F && f)const;

    template<class F>
    void forEachNode(F && f)const;


    // serialization de-serialization

    virtual uint64_t serializationSize() const;

    template<class ITER>
    void serialize(ITER & iter) const;

    template<class ITER>
    void deserialize(ITER & iter);

    UndirectedGraph<EdgeInternalType,NodeInteralType> extractSubgraphFromNodes(
        const std::vector<NODE_INTERNAL_TYPE> & nodeList, 
        std::vector<EDGE_INTERANL_TYPE> & innerEdgesOut,
        std::vector<EDGE_INTERANL_TYPE> & outerEdgesOut) const;

protected:

    bool insertEdgeOnlyInNodeAdj(const int64_t u, const int64_t v);

    template<class PER_THREAD_DATA_VEC>
    void mergeAdjacencies(
        PER_THREAD_DATA_VEC & perThreadDataVec,
        parallel::ThreadPool & threadpool
    );

    std::vector<NodeStorage> nodes_;
    std::vector<EdgeStorage> edges_;
};


template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
UndirectedGraph(const uint64_t numberOfNodes , const uint64_t reserveNumberOfEdges )
:   nodes_(numberOfNodes),
    edges_()
{
    edges_.reserve(reserveNumberOfEdges);
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
void 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
assign(const uint64_t numberOfNodes , const uint64_t reserveNumberOfEdges ){
    nodes_.clear();
    edges_.clear();
    nodes_.resize(numberOfNodes);
    edges_.reserve(reserveNumberOfEdges);
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
int64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
insertEdge(const int64_t u, const int64_t v){

    const auto fres =  nodes_[u].find(NodeAdjacency(v));
    if(fres != nodes_[u].end())
        return fres->edge();
    else{
        const auto uu = std::min(u,v);
        const auto vv = std::max(u,v);
        auto e = EdgeStorage(uu, vv);
        auto ei = edges_.size();
        edges_.push_back(e);
        nodes_[u].insert(NodeAdjacency(v,ei));
        nodes_[v].insert(NodeAdjacency(u,ei));
        return ei;
    }
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
int64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
u(const int64_t e)const{
    NIFTY_ASSERT_OP(e,<,numberOfEdges());
    return edges_[e].first;
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
int64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
v(const int64_t e)const{
    NIFTY_ASSERT_OP(e,<,numberOfEdges());
    return edges_[e].second;
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
int64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
findEdge(const int64_t u, const int64_t v)const{
    NIFTY_ASSERT_OP(u,<,numberOfNodes());
    NIFTY_ASSERT_OP(v,<,numberOfNodes());
    const auto fres =  nodes_[u].find(NodeAdjacency(v));
    if(fres != nodes_[u].end())
        return fres->edge();
    else
        return -1;
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
int64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
nodeIdUpperBound() const{
    return numberOfNodes() == 0 ? 0 : numberOfNodes()-1;
}


template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
int64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
edgeIdUpperBound() const{
    return numberOfEdges() == 0 ? 0 : numberOfEdges()-1;
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
uint64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
numberOfEdges() const {
    return edges_.size();
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
uint64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
numberOfNodes() const{
    return nodes_.size();
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
typename UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::NodeIter 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
nodesBegin()const{
    return NodeIter(0);
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
typename UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::NodeIter 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
nodesEnd()const{
    return NodeIter(this->numberOfNodes());
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
typename UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::EdgeIter 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
edgesBegin()const{
    return EdgeIter(0);
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
typename UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::EdgeIter 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
edgesEnd()const{
    return EdgeIter(this->numberOfEdges());
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
typename UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::AdjacencyIter 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
adjacencyBegin(const int64_t node)const{
    NIFTY_ASSERT_OP(node,<,numberOfNodes());
    return nodes_[node].begin();
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
typename UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::AdjacencyIter 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
adjacencyEnd(const int64_t node)const{
    NIFTY_ASSERT_OP(node,<,numberOfNodes());
    return nodes_[node].end();
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
typename UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::AdjacencyIter 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
adjacencyOutBegin(const int64_t node)const{
    return adjacencyBegin(node);
}


template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
std::pair<int64_t,int64_t> 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
uv(const int64_t e)const{
    const auto _uv = edges_[e];
    return std::pair<int64_t,int64_t>(_uv.first, _uv.second);
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
template<class F>
void 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
forEachEdge(F && f)const{
    for(uint64_t edge=0; edge< numberOfEdges(); ++edge){
        f(edge);
    }
}


template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
template<class F>
void 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
forEachNode(F && f)const{
    for(uint64_t node=0; node< numberOfNodes(); ++node){
        f(node);
    }
}




template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
uint64_t 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
serializationSize() const{
    auto size = 0;
    size += 2; // number of nodes;  number of edges
    size += this->numberOfEdges() * 2;  // u, v; 
    return size;
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
template<class ITER>
void 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
serialize(ITER & iter) const{
    
    *iter = this->numberOfNodes(); 
    ++iter;
    *iter = this->numberOfEdges(); 
    ++iter;

    int i = 0;
    for(const auto edge : this->edges()){
        *iter = this->u(edge);
        ++iter;
        *iter = this->v(edge);
        ++iter;
    }
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
template<class ITER>
void 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
deserialize(ITER & iter){

    const auto nNodes = *iter;
    ++iter;
    const auto nEdges = *iter;
    ++iter;

    this->assign(nNodes, nEdges);

    for(auto e=0; e<nEdges; ++e){
        const auto u = *iter;
        ++iter;
        const auto v = *iter;
        ++iter;
        this->insertEdge(u, v);
    }

}


template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
template<class PER_THREAD_DATA_VEC>
inline void 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
mergeAdjacencies(
    PER_THREAD_DATA_VEC & perThreadDataVec,
    parallel::ThreadPool & threadpool
){
    const auto numberOfLabels = this->numberOfNodes();



    // merge the node adjacency sets for each node
    nifty::parallel::parallel_foreach(threadpool, numberOfLabels, [&](int tid, int label){
        auto & set0 = perThreadDataVec[0].adjacency[label];
        for(size_t i=1; i<perThreadDataVec.size(); ++i){
            const auto & setI = perThreadDataVec[i].adjacency[label];
            set0.insert(setI.begin(), setI.end());
        }
        for(auto otherNode : set0)
             this->nodes_[label].insert(NodeAdjacency(otherNode));
    });

    // insert the edge index for each edge
    uint64_t edgeIndex = 0;
    auto & edges = this->edges_;
    for(uint64_t u = 0; u< numberOfLabels; ++u){
        for(auto & vAdj :  this->nodes_[u]){
            const auto v = vAdj.node();
            if(u < v){
                edges.push_back(EdgeStorage(u, v));
                vAdj.changeEdgeIndex(edgeIndex);
                auto fres =  this->nodes_[v].find(NodeAdjacency(u));
                fres->changeEdgeIndex(edgeIndex);
                ++edgeIndex;
            }
        }
    }
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE >
bool 
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
insertEdgeOnlyInNodeAdj(const int64_t u, const int64_t v){

    const auto fres =  nodes_[u].find(NodeAdjacency(v));
    if(fres != nodes_[u].end())
        return false;
    else{
        nodes_[u].insert(NodeAdjacency(v));
        nodes_[v].insert(NodeAdjacency(u));
        return true;
    }
}

template<class EDGE_INTERANL_TYPE, class NODE_INTERNAL_TYPE>
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>
UndirectedGraph<EDGE_INTERANL_TYPE, NODE_INTERNAL_TYPE>::
extractSubgraphFromNodes(
        const std::vector<NODE_INTERNAL_TYPE> & nodeList, 
        std::vector<EDGE_INTERANL_TYPE> & innerEdgesOut,
        std::vector<EDGE_INTERANL_TYPE> & outerEdgesOut) const {
    
    std::map<NodeInteralType,NodeInteralType> globalToLocalNodes;
    for(size_t i = 0; i < nodeList.size(); ++i)
        globalToLocalNodes.insert( std::make_pair(nodeList[i], NodeInteralType(i)) );

    for(auto u : nodeList) {
        for(auto adjacencyIt = this->adjacencyBegin(u); adjacencyIt != this->adjacencyEnd(u); ++adjacencyIt) {
            auto v = adjacencyIt->node();
            auto e = this->findEdge(u, v);
            if( std::find(nodeList.begin(), nodeList.end(), v) != nodeList.end() )
                innerEdgesOut.push_back(e);
            else
                outerEdgesOut.push_back(e);
        }
    }
    
    // make the edges unique
    std::sort(innerEdgesOut.begin(),innerEdgesOut.end());
    auto last = std::unique(innerEdgesOut.begin(), innerEdgesOut.end());
    innerEdgesOut.erase( last, innerEdgesOut.end() );
    
    std::sort(outerEdgesOut.begin(),outerEdgesOut.end());
    last = std::unique(outerEdgesOut.begin(), outerEdgesOut.end());
    outerEdgesOut.erase( last, outerEdgesOut.end() );
    
    // get number of nodes
    uint64_t numberOfNodes = nodeList.size();
    UndirectedGraph<EdgeInternalType,NodeInteralType> subgraphOut(numberOfNodes);

    // get the local uv - ids
    for(size_t i = 0; i < innerEdgesOut.size(); ++i) {
        auto uv = this->uv(innerEdgesOut[i]);
        subgraphOut.insertEdge(globalToLocalNodes[uv.first], globalToLocalNodes[uv.second]);
    }

    return subgraphOut;
}

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_SIMPLE_GRAPH_HXX
