#pragma once
#ifndef NIFTY_UNDIRECTED_GRAPH_BASE_HXX
#define NIFTY_UNDIRECTED_GRAPH_BASE_HXX

#include <boost/iterator/transform_iterator.hpp>

#include "nifty/graph/graph_maps.hxx"
#include "nifty/tools/const_iterator_range.hxx"


namespace nifty{
namespace graph{





template<
    class CHILD_GRAPH
>
class UndirectedGraphBase{
public:
    typedef CHILD_GRAPH ChildGraph;
    typedef UndirectedGraphBase<ChildGraph> Self;

    template<class T>
    struct NodeMap : graph_maps::NodeMap<ChildGraph,T> {
        using graph_maps::NodeMap<ChildGraph,T>::NodeMap;
    };
    template<class T>
    struct EdgeMap : graph_maps::EdgeMap<ChildGraph,T> {
        using graph_maps::EdgeMap<ChildGraph,T>::EdgeMap;
    };

    template<class _CHILD_GRAPH>
    struct NodeIterRange :  public tools::ConstIteratorRange<typename _CHILD_GRAPH::NodeIter>{
        using tools::ConstIteratorRange<typename _CHILD_GRAPH::NodeIter>::ConstIteratorRange;
    };

    template<class _CHILD_GRAPH>
    struct EdgeIterRange :  public tools::ConstIteratorRange<typename _CHILD_GRAPH::EdgeIter>{
        using tools::ConstIteratorRange<typename _CHILD_GRAPH::EdgeIter>::ConstIteratorRange;
    };

    template<class _CHILD_GRAPH>
    struct AdjacencyIterRange :  public tools::ConstIteratorRange<typename _CHILD_GRAPH::AdjacencyIter>{
        using tools::ConstIteratorRange<typename _CHILD_GRAPH::AdjacencyIter>::ConstIteratorRange;
    };

    // For range based loops over all nodes
    NodeIterRange<ChildGraph > nodes() const{
        return NodeIterRange<ChildGraph>(_child().nodesBegin(),_child().nodesEnd());
    }

    // For range based loops over all edges
    EdgeIterRange<ChildGraph > edges() const{
        return EdgeIterRange<ChildGraph>(_child().edgesBegin(),_child().edgesEnd());
    }

    // For range based loops over adjacency for one node
    AdjacencyIterRange<ChildGraph > adjacency(const int64_t node) const{
        return AdjacencyIterRange<ChildGraph>(_child().adjacencyBegin(node),_child().adjacencyEnd(node));
    }

private:
    ChildGraph & _child(){
       return *static_cast<ChildGraph *>(this);
    }
    const ChildGraph & _child()const{
       return *static_cast<const ChildGraph *>(this);
    }
};




} // namespace nifty::graph
} // namespace nifty
  // 
#endif  // NIFTY_UNDIRECTED_GRAPH_BASE_HXX
