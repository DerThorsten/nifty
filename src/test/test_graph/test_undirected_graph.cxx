#define BOOST_TEST_MODULE NiftyGraphTest

#include <boost/test/unit_test.hpp>

#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/simple_graph.hxx"

BOOST_AUTO_TEST_CASE(UndirectedGraphTest)
{
    auto e = 0;
    nifty::graph::UndirectedGraph<> graph(4);
    NIFTY_TEST_OP(graph.numberOfNodes(),==,4);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,0);

    e = graph.insertEdge(0,1);
    NIFTY_TEST_OP(e,==,0);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,1);
    NIFTY_TEST_OP(graph.u(e),==,0);
    NIFTY_TEST_OP(graph.v(e),==,1);


    e = graph.insertEdge(0,2);
    NIFTY_TEST_OP(e,==,1);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,2);
    NIFTY_TEST_OP(graph.u(e),==,0);
    NIFTY_TEST_OP(graph.v(e),==,2);

    e = graph.insertEdge(0,3);
    NIFTY_TEST_OP(e,==,2);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,3);
    NIFTY_TEST_OP(graph.u(e),==,0);
    NIFTY_TEST_OP(graph.v(e),==,3);


    e = graph.insertEdge(2,3);
    NIFTY_TEST_OP(e,==,3);  
    NIFTY_TEST_OP(graph.numberOfEdges(),==,4);
    NIFTY_TEST_OP(graph.u(e),==,2);
    NIFTY_TEST_OP(graph.v(e),==,3);


    auto c=0;
    for(auto iter = graph.nodesBegin(); iter!=graph.nodesEnd(); ++iter){
        NIFTY_TEST_OP(*iter,==,c);
        ++c;
    }
    NIFTY_TEST_OP(graph.numberOfNodes(),==,c);

    c = 0;
    for(auto node : graph.nodes()){
        NIFTY_TEST_OP(node,==,c);
        ++c;
        for(auto adj : graph.adjacency(node)){
            //std::cout<<"   "<<adj.node()<<"\n";
        }
    }
    NIFTY_TEST_OP(graph.numberOfNodes(),==,c);

    c = 0;
    //std::cout<<"for graph edges\n";
    for(auto & edge : graph.edges()){
        NIFTY_TEST_OP(edge,==,c);
        ++c;
        //std::cout<<edge<<"\n"<<"   "<<graph.u(edge)<<" "<<graph.v(edge)<<"\n";
    }
    NIFTY_TEST_OP(graph.numberOfEdges(),==,c);

}
