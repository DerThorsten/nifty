#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_grid_graph.hxx"

void undirectedGridGraphTest()
{
    //auto e = 0;
    //nifty::graph::UndirectedGraph<> graph(4);
    //NIFTY_TEST_OP(graph.numberOfNodes(),==,4);
    //NIFTY_TEST_OP(graph.numberOfEdges(),==,0);

    typedef nifty::graph::UndirectedGridGraph<2, true> GridGraphType;
    typedef typename GridGraphType::ShapeType ShapeType;


    ShapeType shape({5,4});      
    GridGraphType graph(shape);
    
    NIFTY_TEST_OP(graph.numberOfNodes(),==,shape[0]*shape[1]);
}

int main(){
	undirectedGridGraphTest();
}