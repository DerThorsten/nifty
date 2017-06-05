#pragma once

#include <boost/iterator/transform_iterator.hpp>

#include "nifty/graph/graph_maps.hxx"
#include "nifty/tools/const_iterator_range.hxx"


namespace nifty{
namespace graph{

template<
    class CHILD_GRAPH
>
class DirectedGraphBase{
public:
    typedef CHILD_GRAPH ChildGraph;
    typedef DirectedGraphBase<ChildGraph> Self;

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
    struct ArcIterRange :  public tools::ConstIteratorRange<typename _CHILD_GRAPH::ArcIter>{
        using tools::ConstIteratorRange<typename _CHILD_GRAPH::ArcIter>::ConstIteratorRange;
    };

    // template<class _CHILD_GRAPH>
    // struct EdgeIterRange :  public tools::ConstIteratorRange<typename _CHILD_GRAPH::EdgeIter>{
    //     using tools::ConstIteratorRange<typename _CHILD_GRAPH::EdgeIter>::ConstIteratorRange;
    // };

    template<class _CHILD_GRAPH>
    struct AdjacencyIterRange :  public tools::ConstIteratorRange<typename _CHILD_GRAPH::AdjacencyOutIter>{
        using tools::ConstIteratorRange<typename _CHILD_GRAPH::AdjacencyOutIter>::ConstIteratorRange;
    };

    // For range based loops over all nodes
    NodeIterRange<ChildGraph > nodes() const{
        return NodeIterRange<ChildGraph>(_child().nodesBegin(),_child().nodesEnd());
    }

    // For range based loops over all arcs
    ArcIterRange<ChildGraph > arcs() const{
        return ArcIterRange<ChildGraph>(_child().arcsBegin(),_child().arcsEnd());
    }
    ArcIterRange<ChildGraph > edges() const{
        return ArcIterRange<ChildGraph>(_child().arcsBegin(),_child().arcsEnd());
    }

    // For range based loops over adjacency for one node
    AdjacencyIterRange<ChildGraph > adjacency(const int64_t node) const{
        return AdjacencyIterRange<ChildGraph>(_child().adjacencyOutBegin(node),_child().adjacencyOutEnd(node));
    }

    int64_t findEdge(const uint64_t u, const uint64_t v){
        const auto ef =  _child().findArc(u,v); 
        if(ef != -1) 
            return _child().findArc(u, v);
        else 
            return _child().findArc(v, u);
    }

    int64_t edgeIdUpperBound() const {return _child().maxArcId();}
    int64_t numberOfEdges() const {return _child().numberOfArcs();}
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
