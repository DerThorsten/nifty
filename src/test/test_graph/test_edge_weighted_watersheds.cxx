#define BOOST_TEST_MODULE NiftyEdgeWeightedWatershedsTest

#include <boost/test/unit_test.hpp>

#include <iostream> 
#include <random>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/edge_weighted_watersheds.hxx"


BOOST_AUTO_TEST_CASE(RandomizedEdgeWeightedWatersheds)
{
    // rand gen 
    //std::random_device rd();
    std::mt19937 gen(42);//rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);    



    typedef double WeightType;
    typedef nifty::graph::UndirectedGraph<> Graph;



    // create a grid graph
    const size_t s = 30;
    Graph g(s*s);
    for(auto y=0; y<s; ++y)
    for(auto x=0; x<s; ++x){
        auto u = x + y*s;
        if(x+1 < s){
            auto v = x + 1 + y * s;
            g.insertEdge(u, v);
        }
        if(y+1 < s){
            auto v = x + (y + 1) * s;
            g.insertEdge(u, v);
        }
    }

    std::vector<double> edgeWeights(g.numberOfEdges(),0);
    for(auto edge: g.edges())
        edgeWeights[edge] =  dis(gen);

    std::vector<uint64_t> seeds(g.numberOfNodes(),0);
    std::vector<uint64_t> labels(g.numberOfNodes(),0);

    edgeWeightedWatershedsSegmentation(g, edgeWeights, seeds, labels);

}

