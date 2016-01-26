#pragma once
#ifndef NIFTY_GRAPH_SIMPLE_GRAPH_HXX
#define NIFTY_GRAPH_SIMPLE_GRAPH_HXX

#include <vector>
#include <set>

#include <boost/iterator/counting_iterator.hpp>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_graph_base.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/graph_iterators.hxx"

namespace nifty{
namespace graph{


template<class EDGE_INTERANL_TYPE = int64_t, 
         class NODE_INTERNAL_TYPE = int64_t>
class InsertOnlyGraph : public
    UndirectedGraphBase<
        InsertOnlyGraph<EDGE_INTERANL_TYPE,NODE_INTERNAL_TYPE>,
        boost::counting_iterator<int64_t>,
        boost::counting_iterator<int64_t>
    >
{
private:
    typedef EDGE_INTERANL_TYPE EdgeInternalType;
    typedef NODE_INTERNAL_TYPE NodeInteralType;
    typedef detail_graph::UndirectedAdjacency<int64_t,int64_t,NodeInteralType,EdgeInternalType> NodeAdjacency;
    typedef std::set<NodeAdjacency > NodeStorage;

    typedef std::pair<NodeInteralType,NodeInteralType> EdgeStorage;
public:
    typedef boost::counting_iterator<int64_t> NodeIter;
    typedef boost::counting_iterator<int64_t> EdgeIter;
    typedef typename NodeStorage::const_iterator AdjacencyIter;

    // constructors
    InsertOnlyGraph(const uint64_t numberOfNodes = 0, const uint64_t reserveNumberOfEdges = 0)
    :   nodes_(numberOfNodes),
        edges_()
    {
        edges_.reserve(reserveNumberOfEdges);
    }



    int64_t insertEdge(const int64_t u, const int64_t v){
        auto & adjU = nodes_[u];
        auto & adjV = nodes_[v];    
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

    // MUST IMPL INTERFACE
    int64_t u(const int64_t e)const{
        NIFTY_ASSERT_OP(e,<,numberOfEdges());
        return edges_[e].first;
    }
    int64_t v(const int64_t e)const{
        NIFTY_ASSERT_OP(e,<,numberOfEdges());
        return edges_[e].second;
    }
    int64_t findEdge(const int64_t u, const int64_t v){
        NIFTY_ASSERT_OP(u,<,numberOfNodes());
        NIFTY_ASSERT_OP(v,<,numberOfNodes());
        const auto fres =  nodes_[u].find(NodeAdjacency(v));
        if(fres != nodes_[u].end())
            return fres->edge();
        else
            return -1;
    }
    int64_t maxNodeId() const{return numberOfNodes() == 0 ? -1 : numberOfNodes()-1;}
    int64_t maxEdgeId() const{return numberOfEdges() == 0 ? -1 : numberOfEdges()-1;}
    int64_t numberOfEdges() const {return edges_.size();}
    int64_t numberOfNodes() const{return nodes_.size();}

    NodeIter nodesBegin()const{return NodeIter(0);}
    NodeIter nodesEnd()const{return NodeIter(this->numberOfNodes());}
    EdgeIter edgesBegin()const{return EdgeIter(0);}
    EdgeIter edgesEnd()const{return EdgeIter(this->numberOfEdges());}

    AdjacencyIter adjacencyBegin(const int64_t node)const{
        NIFTY_ASSERT_OP(node,<,numberOfNodes());
        return nodes_[node].begin();
    }
    AdjacencyIter adjacencyEnd(const int64_t node)const{
        NIFTY_ASSERT_OP(node,<,numberOfNodes());
        return nodes_[node].end();
    }

private:



    std::vector<NodeStorage> nodes_;
    std::vector<EdgeStorage> edges_;
};

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_SIMPLE_GRAPH_HXX
