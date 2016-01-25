#define BOOST_TEST_MODULE NiftyShortestPathTest

#include <boost/test/unit_test.hpp>

#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/shortest_path.hxx"

BOOST_AUTO_TEST_CASE(ShortestPathTest)
{
    typedef nifty::graph::InsertOnlyGraph<>  Graph;

    Graph graph(4);

    graph.insertEdge(0,1);
    graph.insertEdge(0,2);
    graph.insertEdge(0,3);
    graph.insertEdge(2,3);

    for(auto edge : graph.edges()){
        std::cout<<edge<<"\n"<<"   "<<graph.u(edge)<<" "<<graph.v(edge)<<"\n";
    }

    for(auto node : graph.nodes()){
        std::cout<<node<<"\n";
        for(auto adj : graph.adjacency(node)){
            std::cout<<"   "<<adj.node()<<"\n";
        }
    }


    typedef nifty::graph::ShortestPathDijkstra<Graph,float> Sp;
    std::vector<float> edgeWeights = {0.1,0.2,0.3,0.4};
    Sp sp(graph);
    sp.run(edgeWeights,{0});
}
