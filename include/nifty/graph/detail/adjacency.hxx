#pragma once
#ifndef NIFTY_UNDIRECTED_GRAPH_DETAIL_ADJACENCY_HXX
#define NIFTY_UNDIRECTED_GRAPH_DETAIL_ADJACENCY_HXX


namespace nifty{
namespace graph{
namespace detail_graph{

// \cond SUPPRESS_DOXYGEN

// an element in the implementation
// of adjacency list
// End users will not notice this class
// => implementation detail
template<
    class NODE_RETURN_TYPE = int64_t,
    class EDGE_OR_ARG_RETURN_TYPE = int64_t,
    class NODE_INTERANL_TYPE = int64_t,
    class EDGE_OR_ARC_INTERNAL_TYPE = int64_t
>
class AdjacencyImpl {
public:
    AdjacencyImpl(const NODE_INTERANL_TYPE node =0, const EDGE_OR_ARC_INTERNAL_TYPE edgeOrArc=0)
    :   node_(node),
        edgeOrArc_(edgeOrArc){
    }
    NODE_RETURN_TYPE  node() const{
        return node_;
    }
    bool operator<(const AdjacencyImpl & other) const{
        return  node_ < other.node_;
    }

    void changeEdgeIndex(const EDGE_OR_ARC_INTERNAL_TYPE newEdge)const{
        edgeOrArc_ = newEdge;
    }
protected:
    EDGE_OR_ARG_RETURN_TYPE  edgeOrArc() const{
        return edgeOrArc_;
    }
private:
    NODE_INTERANL_TYPE node_;
    mutable EDGE_OR_ARC_INTERNAL_TYPE edgeOrArc_;
};


template<
    class NODE_RETURN_TYPE = int64_t,
    class EDGE_RETURN_TYPE = int64_t,
    class NODE_INTERANL_TYPE = int64_t,
    class EDGE_INTERNAL_TYPE = int64_t
>
class UndirectedAdjacency : public AdjacencyImpl<NODE_RETURN_TYPE,EDGE_RETURN_TYPE,NODE_INTERANL_TYPE,EDGE_INTERNAL_TYPE>
{
public:
    UndirectedAdjacency(const NODE_INTERANL_TYPE node =0, const EDGE_INTERNAL_TYPE edge=0)
    :   AdjacencyImpl<NODE_RETURN_TYPE,EDGE_RETURN_TYPE,NODE_INTERANL_TYPE,EDGE_INTERNAL_TYPE>(node,edge){
    }
    EDGE_RETURN_TYPE edge()const{
        return this->edgeOrArc();
    }
};

template<
    class NODE_RETURN_TYPE = int64_t,
    class ARC_RETURN_TYPE = int64_t,
    class NODE_INTERANL_TYPE = int64_t,
    class ARC_INTERNAL_TYPE = int64_t
>
class DirectedAdjacency : public AdjacencyImpl<NODE_RETURN_TYPE,ARC_RETURN_TYPE,NODE_INTERANL_TYPE,ARC_INTERNAL_TYPE>
{
public:
    DirectedAdjacency(const NODE_INTERANL_TYPE node =0, const ARC_INTERNAL_TYPE arc=0)
    :   AdjacencyImpl<NODE_RETURN_TYPE,ARC_RETURN_TYPE,NODE_INTERANL_TYPE,ARC_INTERNAL_TYPE>(node,arc){
    }
    ARC_RETURN_TYPE arc()const{
        return this->edgeOrArc();
    }
    ARC_RETURN_TYPE edge()const{
        return this->edgeOrArc();
    }
};

// \endcond SUPPRESS_DOXYGEN


} // namespace nifty::graph::detail_graph
} // namespace nifty::graph
} // namespace nifty
  // 
#endif  // NIFTY_UNDIRECTED_GRAPH_DETAIL_ADJACENCY_HXX
