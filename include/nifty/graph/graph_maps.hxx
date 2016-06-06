#pragma once
#ifndef NIFTY_UNDIRECTED_GRAPH_GRAPH_MAPS_HXX
#define NIFTY_UNDIRECTED_GRAPH_GRAPH_MAPS_HXX

namespace nifty{
namespace graph{
namespace graph_maps{

template<class G, class T>
struct NodeMap : public std::vector<T>{
    NodeMap( const G & g, const T & val)
    :   std::vector<T>( g.maxNodeId()+1, val){
    }
    NodeMap( const G & g)
    :   std::vector<T>( g.maxNodeId()+1){
    }
    NodeMap( )
    :   std::vector<T>( ){
    }
};
template<class G, class T>
struct EdgeMap : public std::vector<T>{
    EdgeMap( const G & g, const T & val)
    :   std::vector<T>( g.maxEdgeId()+1, val){
    }

    EdgeMap( const G & g)
    :   std::vector<T>( g.maxEdgeId()+1){
    }
    
    EdgeMap( )
    :   std::vector<T>( ){
    }
};






} // namespace nifty::graph::graph_maps
} // namespace nifty::graph
} // namespace nifty
  // 
#endif  // NIFTY_UNDIRECTED_GRAPH_GRAPH_MAPS_HXX
