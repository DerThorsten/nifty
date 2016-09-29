#include "examples_common.hxx"

#include <iostream>
#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

int main( int argc , char *argv[] ){

    // nifty::graph::UndirectedGraph has contiguous
    // edge and node ids/indices which start at zero
    // 
    // graph with 5 nodes and 6 edges
    // 
    //   0 - 1
    //   |   | \ 
    //   2 - 3 - 4
    // 
    // 
    nifty::graph::UndirectedGraph<> graph(5);

    // inserte edges 
    auto e0 = graph.insertEdge(0,1);  
    auto e1 = graph.insertEdge(0,2);    
    auto e2 = graph.insertEdge(1,3);    
    auto e3 = graph.insertEdge(1,4);    
    auto e4 = graph.insertEdge(2,3);    
    auto e5 = graph.insertEdge(3,4);    

    
    // loop over all nodes with auto range
    // node is an uint64_t type
    for(const auto node : graph.nodes()){
        std::cout<<"Node "<<node<<"\n";
    }

    // do something for each node with a lambda
    graph.forEachNode([](const uint64_t node){
        std::cout<<"Node "<<node<<"\n";
    });
    

    // loop over all edges with auto range
    // edge is an uint64_t type
    for(const auto edge : graph.edges()){
        std::cout<<"Edge "<<edge<<"\n";
    }
    
    // do something for each edge with a lambda
    graph.forEachEdge([](const uint64_t edge){
        std::cout<<"Edge "<<edge<<"\n";
    });

    return 0;
}