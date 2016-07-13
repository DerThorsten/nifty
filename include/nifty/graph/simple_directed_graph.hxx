#pragma once
#ifndef NIFTY_GRAPH_SIMPLE_UNDIRECTED_GRAPH_HXX
#define NIFTY_GRAPH_SIMPLE_UNDIRECTED_GRAPH_HXX

#include <vector>
#include <set>

#include <boost/iterator/counting_iterator.hpp>

#include "nifty/graph/directed_graph_base.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/graph_tags.hxx"

namespace nifty{
namespace graph{


template<class ARC_INTERANL_TYPE = int64_t, 
         class NODE_INTERNAL_TYPE = int64_t>
class SimpleDirectedGraph : public DirectedGraphBase<
    SimpleDirectedGraph<ARC_INTERANL_TYPE, NODE_INTERNAL_TYPE>
>
{
private:
    typedef ARC_INTERANL_TYPE ArcInternalType;
    typedef NODE_INTERNAL_TYPE NodeInteralType;
    typedef detail_graph::DirectedAdjacency<int64_t,int64_t,NodeInteralType,ArcInternalType> NodeAdjacency;
    typedef std::set<NodeAdjacency > DirectedNodeStorage;
    typedef std::pair<DirectedNodeStorage, DirectedNodeStorage> NodeStorage;
    typedef std::pair<NodeInteralType,NodeInteralType> ArcStorage;
public:
    typedef boost::counting_iterator<int64_t> NodeIter;
    typedef boost::counting_iterator<int64_t> ArcIter;

    typedef typename DirectedNodeStorage::const_iterator AdjacencyInIter;
    typedef typename DirectedNodeStorage::const_iterator AdjacencyOutIter;
    typedef ContiguousTag ArcIdTag;
    typedef ContiguousTag EdgeIdTag;
    typedef ContiguousTag NodeIdTag;
    // constructors
    SimpleDirectedGraph(const uint64_t numberOfNodes = 0, const uint64_t reserveNumberOfArcs = 0)
    :   nodes_(numberOfNodes),
        arcs_()
    {
        arcs_.reserve(reserveNumberOfArcs);
    }

    int64_t insertArc(const int64_t s, const int64_t t){
        // from s to t
        auto & sOut = nodes_[s].first;
        auto & tIn  = nodes_[t].second;
        const auto fRes = sOut.find(NodeAdjacency(t));
        if(fRes == sOut.end()){
            auto arc = arcs_.size();
            sOut.insert(NodeAdjacency(t,arc));
            tIn.insert(NodeAdjacency(s,arc));
            arcs_.push_back(ArcStorage(s,t));
            return arc;
        }
        else{
            return fRes->arc();
        }
    }

    // MUST IMPL INTERFACE
    int64_t source(const int64_t a)const{
        NIFTY_ASSERT_OP(a,<,numberOfArcs());
        return arcs_[a].first;
    }
    int64_t target(const int64_t a)const{
        NIFTY_ASSERT_OP(a,<,numberOfArcs());
        return arcs_[a].second;
    }



    int64_t findArc(const int64_t s, const int64_t t){
        auto & sOut = nodes_[s].first;
        auto & tIn  = nodes_[t].second;
        const auto fRes = sOut.find(NodeAdjacency(t));
        return fRes == sOut.end() ? -1 : fRes->arc();
    }

    int64_t nodeIdUpperBound() const{return numberOfNodes() == 0 ? -1 : numberOfNodes()-1;}
    int64_t maxArcId() const{return numberOfArcs() == 0 ? -1 : numberOfArcs()-1;}
    int64_t numberOfArcs() const {return arcs_.size();}
    int64_t numberOfNodes() const{return nodes_.size();}


    NodeIter nodesBegin()const{return NodeIter(0);}
    NodeIter nodesEnd()const{return NodeIter(this->numberOfNodes());}
    ArcIter arcsBegin()const{return ArcIter(0);}
    ArcIter arcsEnd()const{return ArcIter(this->numberOfArcs());}

    AdjacencyOutIter adjacencyOutBegin(const int64_t node)const{
        NIFTY_ASSERT_OP(node,<,numberOfNodes());
        return nodes_[node].first.begin();
    }
    AdjacencyOutIter adjacencyOutEnd(const int64_t node)const{
        NIFTY_ASSERT_OP(node,<,numberOfNodes());
        return nodes_[node].first.end();
    }

    AdjacencyInIter adjacencyInBegin(const int64_t node)const{
        NIFTY_ASSERT_OP(node,<,numberOfNodes());
        return nodes_[node].second.begin();
    }
    AdjacencyInIter adjacencyInEnd(const int64_t node)const{
        NIFTY_ASSERT_OP(node,<,numberOfNodes());
        return nodes_[node].second.end();
    }


private:
    std::vector<NodeStorage> nodes_;
    std::vector<ArcStorage> arcs_;
};

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_SIMPLE_UNDIRECTED_GRAPH_HXX
