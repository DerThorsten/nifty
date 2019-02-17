#include <iostream> 
#include <random>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/graph/opt/multicut/multicut_ilp.hxx"

#ifdef WITH_GUROBI
#include "nifty/ilp_backend/gurobi.hxx"
#endif

#ifdef WITH_CPLEX
#include "nifty/ilp_backend/cplex.hxx"
#endif

#ifdef WITH_GLPK
#include "nifty/ilp_backend/glpk.hxx"
#endif

void randomizedMulticutTest()
{
    // rand gen 
    //std::random_device rd();
    std::mt19937 gen(42);//rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);    



    typedef double WeightType;
    typedef nifty::graph::UndirectedGraph<> GraphType;
    typedef nifty::graph::opt::multicut::MulticutObjective<GraphType, WeightType> ObjectiveType;
    typedef nifty::graph::opt::multicut::MulticutVerboseVisitor<ObjectiveType> VerboseVisitor;



    // create a grid graph
    const std::size_t s = 15;
    GraphType g(s*s);
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


    // create an objective 
    ObjectiveType objective(g);
    auto & weights = objective.weights(); 

    // fill the objective with values
    for(auto e : g.edges())
        weights[e] =  dis(gen);

    // optimize gurobi
    #ifdef WITH_GUROBI
    {
        std::cout<<"opt gurobi \n";
        typedef nifty::ilp_backend::Gurobi IlpSolver;
        typedef nifty::graph::opt::multicut::MulticutIlp<ObjectiveType, IlpSolver> Solver;
        typedef typename Solver::NodeLabelsType NodeLabelsType;
        // optimize 
        Solver solver(objective);
        nifty::graph::graph_maps::EdgeMap<GraphType, uint8_t> outputEdgeLabels(g,0);
        
        VerboseVisitor visitor; 
        NodeLabelsType nodeLabels(g, 0);
        solver.optimize(nodeLabels, &visitor);
        g.nodeLabelsToEdgeLabels(nodeLabels, outputEdgeLabels);     
    }
    #endif

    #ifdef WITH_CPLEX
    {
        std::cout<<"opt cplex \n";
        typedef nifty::ilp_backend::Cplex IlpSolver;
        typedef nifty::graph::opt::multicut::MulticutIlp<ObjectiveType, IlpSolver> Solver;
        typedef typename Solver::NodeLabelsType NodeLabelsType;
        // optimize 
        Solver solver(objective);
        nifty::graph::graph_maps::EdgeMap<GraphType, uint8_t> outputEdgeLabels(g,0);
        
        VerboseVisitor visitor; 
        NodeLabelsType nodeLabels(g, 0);
        solver.optimize(nodeLabels, &visitor);
        g.nodeLabelsToEdgeLabels(nodeLabels, outputEdgeLabels);     
    }
    #endif

    #ifdef WITH_GLPK
    {
        std::cout<<"opt glpk \n";
        typedef nifty::ilp_backend::Glpk IlpSolver;
        typedef nifty::graph::opt::multicut::MulticutIlp<ObjectiveType, IlpSolver> Solver;
        typedef typename Solver::NodeLabelsType NodeLabelsType;
        // optimize 
        Solver solver(objective);
        nifty::graph::graph_maps::EdgeMap<GraphType, uint8_t> outputEdgeLabels(g,0);
        
        VerboseVisitor visitor; 
        NodeLabelsType nodeLabels(g, 0);
        solver.optimize(nodeLabels, &visitor);
        g.nodeLabelsToEdgeLabels(nodeLabels, outputEdgeLabels);     
    }
    #endif
}


void simpleMulticutTest()
{
    // rand gen 
    //std::random_device rd();
    std::mt19937 gen(42);//rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);    



    typedef double WeightType;
    typedef nifty::graph::UndirectedGraph<> GraphType;
    typedef nifty::graph::opt::multicut::MulticutObjective<GraphType, WeightType> ObjectiveType;
    typedef nifty::graph::opt::multicut::MulticutVerboseVisitor<ObjectiveType> VerboseVisitor;




    // create a grid graph
    const std::size_t sx = 8;
    const std::size_t sy = 13;
    auto node = [&](uint64_t x, uint64_t y){
        return x + y*sx;
    };
    GraphType g(sx*sy);
    for(auto y=0; y<sy; ++y)
    for(auto x=0; x<sx; ++x){
        auto u = node(x, y);
        if(x+1 < sx){
            auto v = node(x+1, y);
            g.insertEdge(u, v);
        }
        if(y+1 < sy){
            auto v = node(x, y+1);
            g.insertEdge(u, v);
        }
    }


    // create an objective 
    ObjectiveType objective(g);
    auto & weights = objective.weights(); 

    // fill the objective with values
    for(auto e : g.edges())
        weights[e] =  10.0;

    auto yPos = std::vector<int>({0, 2, 4, 8, 10,11, 12});
    for(auto y : yPos){
        auto e = g.findEdge( node(3, y), node(4, y)  );
        NIFTY_ASSERT_OP(e,!=,-1);
        weights[e] = -10.0;
    }

    typename GraphType::EdgeMap<uint16_t> shouldSolution(g,0);
    for(std::size_t y=0; y<sy; ++y){
        auto e = g.findEdge( node(3, y), node(3+1, y)  );
        shouldSolution[e] = 1;
    }



    // optimize gurobi
    #ifdef WITH_GUROBI
    {
        std::cout<<"opt gurobi \n";
        typedef nifty::ilp_backend::Gurobi IlpSolver;
        typedef nifty::graph::opt::multicut::MulticutIlp<ObjectiveType, IlpSolver> Solver;
        
        typedef typename Solver::NodeLabelsType NodeLabelsType;
        Solver solver(objective);

        nifty::graph::graph_maps::EdgeMap<GraphType, uint16_t> outputEdgeLabels(g,0);

        VerboseVisitor visitor; 
        NodeLabelsType nodeLabels(g, 0);
        solver.optimize(nodeLabels, &visitor);
        g.nodeLabelsToEdgeLabels(nodeLabels, outputEdgeLabels);

        g.nodeLabelsToEdgeLabels(nodeLabels, outputEdgeLabels);
        for(auto e : g.edges()){
            NIFTY_TEST_OP(shouldSolution[e],==,outputEdgeLabels[e]);
        }
    }
    #endif

    // optimize cplex
    #ifdef WITH_CPLEX
    {
        std::cout<<"opt cplex \n";
        typedef nifty::ilp_backend::Cplex IlpSolver;
        typedef nifty::graph::opt::multicut::MulticutIlp<ObjectiveType, IlpSolver> Solver;
        typedef typename Solver::NodeLabelsType NodeLabelsType;


        Solver solver(objective);
        nifty::graph::graph_maps::EdgeMap<GraphType, uint16_t> outputEdgeLabels(g,0);



        VerboseVisitor visitor; 
        NodeLabelsType nodeLabels(g, 0);
        solver.optimize(nodeLabels, &visitor);
        g.nodeLabelsToEdgeLabels(nodeLabels, outputEdgeLabels);
        
        for(auto e : g.edges()){
            NIFTY_TEST_OP(shouldSolution[e],==,outputEdgeLabels[e]);
        }
    }
    #endif

    // optimize cplex
    #ifdef WITH_GLPK
    {
        std::cout<<"opt glpk \n";
        typedef nifty::ilp_backend::Glpk IlpSolver;
        typedef nifty::graph::opt::multicut::MulticutIlp<ObjectiveType, IlpSolver> Solver;
        typedef typename Solver::NodeLabelsType NodeLabelsType;


        Solver solver(objective);
        nifty::graph::graph_maps::EdgeMap<GraphType, uint16_t> outputEdgeLabels(g,0);



        VerboseVisitor visitor; 
        NodeLabelsType nodeLabels(g, 0);
        solver.optimize(nodeLabels, &visitor);
        g.nodeLabelsToEdgeLabels(nodeLabels, outputEdgeLabels);
        
        for(auto e : g.edges()){
            NIFTY_TEST_OP(shouldSolution[e],==,outputEdgeLabels[e]);
        }
    }
    #endif
}

int main(){
    randomizedMulticutTest();
    simpleMulticutTest();
}
