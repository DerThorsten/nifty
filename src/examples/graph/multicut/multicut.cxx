#include "examples_common.hxx"

#include <iostream>
#include "nifty/tools/runtime_check.hxx"

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/graph/optimization/multicut/multicut_greedy_additive.hxx"

int main( int argc , char *argv[] ){

    

    // nifty::graph::UndirectedGraph has contiguous
    // edge and node ids/indices which start at zero
    // 
    // graph with 6 nodes and 7 edges
    // 
    //   0 - 1 - 2
    //   |   |   |
    //   3 - 4 - 5
    // 
    // 
    typedef nifty::graph::UndirectedGraph<> Graph;
    Graph graph(6);

    // inserte edges 
    graph.insertEdge(0,1);  
    graph.insertEdge(1,2);    
    graph.insertEdge(3,4);    
    graph.insertEdge(4,5);    
    graph.insertEdge(0,3);   
    graph.insertEdge(1,4);   
    graph.insertEdge(2,5);   

    // create multicut objective
    // do not add more edges to graph after creating the mc objective
    typedef nifty::graph::MulticutObjective<Graph, float> MulticutObjective;
    MulticutObjective objective(graph);

    // Set edge weights 
    auto & weights = objective.weights();
    weights[graph.findEdge(0,1)] =  1.0;
    weights[graph.findEdge(1,2)] =  1.0;
    weights[graph.findEdge(3,4)] =  1.0;
    weights[graph.findEdge(4,5)] =  1.0;
    weights[graph.findEdge(0,3)] = -1.0;
    weights[graph.findEdge(1,4)] =  0.5;
    weights[graph.findEdge(2,5)] = -1.0;


    // solve not very powerful but simple solver
    {  
       
        typedef nifty::graph::MulticutGreedyAdditive<MulticutObjective> MulticutSolver;
        typedef nifty::graph::MulticutVerboseVisitor<MulticutObjective> MulticutVerboseVisitor;
        typedef typename MulticutSolver::NodeLabels NodeLabels;
        MulticutSolver solver(objective);

        NodeLabels labels(graph);
        MulticutVerboseVisitor visitor;
        solver.optimize(labels, &visitor);

        std::cout<<"isEdgeCut?\n";
        for(const auto edge : graph.edges()){

            const auto uv = graph.uv(edge);
            const auto u = uv.first;
            const auto v = uv.second;
            const auto isCut = int(labels[u] != labels[v]);
            const auto w = objective.weights()[edge];

            std::cout<<"edge "<<edge<<" iscut? "<< isCut<<"\n";
        }
    }



}