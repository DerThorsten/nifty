#pragma once
#ifndef NIFTY_UNDIRECTED_GRAPH_GRAPH_MAPS_HXX
#define NIFTY_UNDIRECTED_GRAPH_GRAPH_MAPS_HXX

#include "nifty/marray/marray.hxx"

namespace nifty{
namespace graph{
namespace graph_maps{




template<class G, class T>
struct NodeMap : public std::vector<T>{
    NodeMap( const G & g, const T & val)
    :   std::vector<T>( g.nodeIdUpperBound()+1, val){
    }
    NodeMap( const G & g)
    :   std::vector<T>( g.nodeIdUpperBound()+1){
    }
    NodeMap( )
    :   std::vector<T>( ){
    }

    // graph has been modified
    void insertedNodes(const uint64_t nodeId, const T & insertValue = T()){
        if(nodeId == this->size()){
            this->push_back(insertValue);
        }
        else if(nodeId > this->size()){
            this->resize(nodeId + 1, insertValue);
        }
    }
};

template<class G, class T>
struct EdgeMap : public std::vector<T>{
    EdgeMap( const G & g, const T & val)
    :   std::vector<T>( g.edgeIdUpperBound()+1, val){
    }

    EdgeMap( const G & g)
    :   std::vector<T>( g.edgeIdUpperBound()+1){
    }
    
    EdgeMap( )
    :   std::vector<T>( ){
    }

    // graph has been modified
    void insertedEdges(const uint64_t edgeId, const T & insertValue = T()){
        if(edgeId == this->size()){
            this->push_back(insertValue);
        }
        else if(edgeId > this->size()){
            this->resize(edgeId + 1, insertValue);
        }
    }
};










} // namespace nifty::graph::graph_maps
} // namespace nifty::graph
} // namespace nifty
  // 
#endif  // NIFTY_UNDIRECTED_GRAPH_GRAPH_MAPS_HXX
