#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/shortest_path_dijkstra.hxx"
#include "nifty/graph/depth_first_search.hxx"

void depthFirstSearchTest()
{
    typedef nifty::graph::UndirectedGraph<>  Graph;
    Graph g(6);

    //   0 | 1 |
    //   _   _ 
    //   2 | 3 | 4 | 5
    
    auto e01 = g.insertEdge(0,1);
    auto e02 = g.insertEdge(0,2);
    auto e13 = g.insertEdge(1,3);
    auto e23 = g.insertEdge(2,3);
    auto e34 = g.insertEdge(3,4);
    auto e45 = g.insertEdge(4,5);
   

    typedef nifty::graph::DepthFirstSearch<Graph> Bfs;

    // single source single target
    {
        Bfs pf(g);
        pf.runSingleSourceSingleTarget(0,1);

        const auto & pmap = pf.predecessors();
        const auto & dmap = pf.distances();

        NIFTY_TEST_OP(pmap[0],==,0);
        NIFTY_TEST_OP(pmap[1],==,0);
        //NIFTY_TEST_OP(pmap[2],==,-1);
        //NIFTY_TEST_OP(pmap[3],==,-1);
        //NIFTY_TEST_OP(pmap[4],==,-1);
        //NIFTY_TEST_OP(pmap[5],==,-1);

        NIFTY_TEST_OP(dmap[0],==,0);
        NIFTY_TEST_OP(dmap[1],==,1);
        //NIFTY_TEST_EQ(dmap[2],1);
        //NIFTY_TEST_EQ(dmap[3],2);
        //NIFTY_TEST_EQ(dmap[4],3);
        //NIFTY_TEST_EQ(dmap[5],4);

    }

    // single source to all 
    {
        Bfs pf(g);
        pf.runSingleSource(0);

        const auto & pmap = pf.predecessors();
        const auto & dmap = pf.distances();

        NIFTY_TEST_OP(pmap[0],==,0);
        NIFTY_TEST_OP(pmap[1],==,0);
        NIFTY_TEST_OP(pmap[2],==,0);
        NIFTY_TEST(pmap[3]==2 || pmap[3]==1);
        NIFTY_TEST_OP(pmap[4],==,3);
        NIFTY_TEST_OP(pmap[5],==,4);

        NIFTY_TEST_OP(dmap[0],==,0);
        NIFTY_TEST_OP(dmap[1],==,1);
        NIFTY_TEST_OP(dmap[2],==,1);
        NIFTY_TEST_OP(dmap[3],==,2);
        NIFTY_TEST_OP(dmap[4],==,3);
        NIFTY_TEST_OP(dmap[5],==,4);
    }

}

int main(){
    depthFirstSearchTest();
}