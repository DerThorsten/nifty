#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/simple_directed_graph.hxx"
#include "nifty/graph/shortest_path_bellman_ford.hxx"


void shortestPathBellmanFordTest()
{
    typedef nifty::graph::SimpleDirectedGraph<>  Graph;
    Graph g(6);

    //   0 | 1 |
    //   _   _ 
    //   2 | 3 | 4 | 5
    
    auto e01 = g.insertArc(0,1);
    auto e02 = g.insertArc(0,2);
    auto e13 = g.insertArc(3,1);
    auto e23 = g.insertArc(2,3);
    auto e34 = g.insertArc(3,4);
    auto e45 = g.insertArc(4,5);
    std::vector<float> ew = {10.0,2.0,3.0,4.0,20.0,1.0};

    typedef nifty::graph::ShortestPathBellmanFord<Graph,float> Sp;

    NIFTY_TEST_OP(g.numberOfEdges(),==,6);
    NIFTY_TEST_OP(g.numberOfNodes(),==,6);

    // single source to all 
    {
        Sp pf(g);
        pf.runSingleSource(ew,0);

        const auto & pmap = pf.predecessors();
        const auto & dmap = pf.distances();



        NIFTY_TEST_EQ_TOL(dmap[0],0.0f,0.00001);
        NIFTY_TEST_EQ_TOL(dmap[2],2.0f,0.00001);
        NIFTY_TEST_EQ_TOL(dmap[3],6.0f,0.00001);
        NIFTY_TEST_EQ_TOL(dmap[1],9.0f,0.00001);
        NIFTY_TEST_EQ_TOL(dmap[4],26.0f,0.00001);
        NIFTY_TEST_EQ_TOL(dmap[5],27.0f,0.00001);

        NIFTY_TEST_OP(pmap[1],==,3);
        NIFTY_TEST_OP(pmap[3],==,2);
        NIFTY_TEST_OP(pmap[2],==,0);
        NIFTY_TEST_OP(pmap[0],==,0);
        NIFTY_TEST_OP(pmap[4],==,3);
        NIFTY_TEST_OP(pmap[5],==,4);

    }

}

int main(){
    shortestPathBellmanFordTest();
}
