#pragma once
#ifndef NIFTY_UNDIRECTED_GRAPH_DETAIL_ADJACENCY_HXX
#define NIFTY_UNDIRECTED_GRAPH_DETAIL_ADJACENCY_HXX


namespace nifty{
namespace graph{
namespace detail_graph{

// an element in the implementation
// of adjacency list
// End users will not notice this class
// => implementation detail
template<
    class NODE_RETURN_TYPE = int64_t,
    class EDGE_RETURN_TYPE = int64_t,
    class NODE_INTERANL_TYPE = int64_t,
    class EDGE_INTERANL_TYPE = int64_t
>
class Adjacency {
public:
    Adjacency(const NODE_INTERANL_TYPE node =0, const EDGE_INTERANL_TYPE edge=0)
    :   node_(node),
        edge_(edge){
    }
    NODE_RETURN_TYPE  node() const{
        return node_;
    }
    EDGE_RETURN_TYPE  edge() const{
        return edge_;
    }
    bool operator<(const Adjacency & other) const{
        return  node_ < other.node_;
    }
private:
    NODE_INTERANL_TYPE node_;
    EDGE_INTERANL_TYPE edge_;
};





} // namespace nifty::graph::detail_graph
} // namespace nifty::graph
} // namespace nifty
  // 
#endif  // NIFTY_UNDIRECTED_GRAPH_DETAIL_ADJACENCY_HXX
