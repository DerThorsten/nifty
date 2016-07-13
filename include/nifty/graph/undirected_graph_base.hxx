#pragma once
#ifndef NIFTY_UNDIRECTED_GRAPH_BASE_HXX
#define NIFTY_UNDIRECTED_GRAPH_BASE_HXX

#include <boost/iterator/transform_iterator.hpp>

#include "nifty/graph/graph_tags.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/tools/const_iterator_range.hxx"


namespace nifty{
namespace graph{

namespace detail_undirected_graph_base{



template<class GRAPH, class TAG>
struct GraphItemGeneralization;

template<class GRAPH>
struct GraphItemGeneralization<GRAPH, EdgeTag>{

    typedef typename GRAPH::ChildGraph ChildGraph;
    typedef typename GRAPH:: template EdgeIterRange<ChildGraph> type;


    static type items(const ChildGraph & g){
        return g.edges();
    }

    static uint64_t numberOfItems(const ChildGraph & g){
        return g.numberOfEdges();
    }

};

template<class GRAPH>
struct GraphItemGeneralization<GRAPH, NodeTag>{
    typedef typename GRAPH::ChildGraph ChildGraph;
    typedef typename GRAPH:: template NodeIterRange<ChildGraph> type;

    static type items(const ChildGraph & g){
        return g.nodes();
    }

    static uint64_t numberOfItems(const ChildGraph & g){
        return g.numberOfNodes();
    }
};

}


template<
    class CHILD_GRAPH,
    class NODE_ITER,
    class EDGE_ITER,
    class ADJACENCY_ITER
>
class UndirectedGraphBase{
public:


    typedef CHILD_GRAPH ChildGraph;
    typedef UndirectedGraphBase<ChildGraph, NODE_ITER, EDGE_ITER, ADJACENCY_ITER> Self;



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

    template<class TAG>
    typename detail_undirected_graph_base::GraphItemGeneralization<Self,TAG>::type items()const{
        return detail_undirected_graph_base::GraphItemGeneralization<Self,TAG>::items(_child());
    }

    // For range based loops over adjacency for one node
    AdjacencyIterRange<ChildGraph > adjacency(const int64_t node) const{
        return AdjacencyIterRange<ChildGraph>(_child().adjacencyBegin(node),_child().adjacencyEnd(node));
    }
    AdjacencyIterRange<ChildGraph > adjacencyIn(const int64_t node) const{
        return _child().adjacency(node);
    }
    AdjacencyIterRange<ChildGraph > adjacencyOut(const int64_t node) const{
        return _child().adjacency(node);
    }
    ADJACENCY_ITER adjacencyOutBegin(const int64_t node)const{
        return _child().adjacencyBegin(node);
    }
    ADJACENCY_ITER adjacencyOutEnd(const int64_t node)const{
        return _child().adjacencyEnd(node);
    }

    ADJACENCY_ITER adjacencyInBegin(const int64_t node)const{
        return _child().adjacencyBegin(node);
    }
    ADJACENCY_ITER adjacencyInEnd(const int64_t node)const{
        return _child().adjacencyEnd(node);
    }

    std::pair<int64_t,int64_t> uv(const int64_t edge)const{
        const auto u = _child().u(edge);
        const auto v = _child().v(edge);
        return std::pair<int64_t, int64_t>(u, v);
    }

    template<class NODE_LABELS, class EDGE_LABELS>
    void nodeLabelsToEdgeLabels(const NODE_LABELS & nodeLabels, EDGE_LABELS & edgeLabels){
        _child().forEachEdge([&](const int64_t edge){
            const auto uv = _child().uv(edge);
            edgeLabels[edge] = nodeLabels[uv.first] != nodeLabels[uv.second] ? 1 : 0;
        });
    }


    template<class F>
    void forEachEdge(F && f)const{
        for(auto edge : _child().edges()){
            f(edge);
        }
    }

    template<class F>
    void forEachNode(F && f)const{
        for(auto node : _child().nodes()){
            f(node);
        }
    }

    template<class F, class TAG>
    void forEachItem(F && f)const{
        for(auto item : _child(). template items<TAG>()){
            f(item);
        }
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
