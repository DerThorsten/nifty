#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_ILP_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_ILP_HXX


#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/three_cycles.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/graph/bidirectional_breadth_first_search.hxx"

namespace nifty{
namespace graph{


    template<class OBJECTIVE, class ILP_SOLVER>
    class MulticutIlp{
    public: 

        typedef OBJECTIVE Objective;
        typedef ILP_SOLVER IlpSovler;
        typedef typename Objective::Graph Graph;
    private:
        typedef ComponentsUfd<Graph> Components;


        struct SubgraphWithCut {
            SubgraphWithCut(const IlpSovler& ilpSolver)
                : ilpSolver_(ilpSolver) 
            {}
            bool useNode(const size_t v) const
                { return true; }
            bool useEdge(const size_t e) const
                { return ilpSolver_.label(e) == 0; }
            const IlpSovler& ilpSolver_;
        };

    public:

        struct Settings{

            size_t numberOfIterations{0};
            bool verbose { true };
            bool addThreeCyclesConstraints{true};

            
        };


        MulticutIlp(const Objective & objective, const Settings & settings = Settings())
        :   objective_(objective),
            graph_(objective.graph()),
            ilpSolver_(),
            components_(graph_),
            settings_(settings),
            variables_(   std::max(uint64_t(3),uint64_t(graph_.numberOfEdges()))),
            coefficients_(std::max(uint64_t(3),uint64_t(graph_.numberOfEdges())))
        {
            this->initializeIlp(ilpSolver_);

            ilpSolver_.setVerbosity(settings_.verboseIlp);

            // add explicit constraints
            if(settings_.addThreeCyclesConstraints){
                this->addThreeCyclesConstraintsExplicitly();
            }
        }



        template<class OUTPUT_EDGEE_LABLES>
        void optimize(OUTPUT_EDGEE_LABLES & outputEdgeLabels){


            

            ilpSolver_.setStart(outputEdgeLabels.begin());


            std::vector<size_t> variables(   std::max(uint64_t(3),uint64_t(graph_.numberOfEdges())));
            std::vector<double> coefficients(std::max(uint64_t(3),uint64_t(graph_.numberOfEdges())));



            //BreadthFirstSearch<Graph> bfs(graph_);
            BidirectionalBreadthFirstSearch<Graph> bibfs(graph_);

            auto addCycleInequalities = [&] (){

                components_.build(SubgraphWithCut(ilpSolver_));

                // search for violated non-chordal cycles and add corresp. inequalities
                size_t nCycle = 0;

                for (size_t edge = 0; edge < graph_.numberOfEdges(); ++edge){
                    if (ilpSolver_.label(edge) > 0.5){

                        auto v0 = graph_.u(edge);
                        auto v1 = graph_.v(edge);

                        if (components_.areConnected(v0, v1)){   

                            bibfs.runSingleSourceSingleTarget(v0, v1, SubgraphWithCut(ilpSolver_));
                            const auto & path = bibfs.path();
                            const auto sz = path.size(); //buildPathInLargeEnoughBuffer(v0, v1, bfs.predecessors(), path.begin());

                            if (findChord(graph_, path.begin(), path.end(), true) != -1)
                                continue;

                            for (size_t j = 0; j < sz - 1; ++j){
                                variables_[j] = graph_.findEdge(path[j], path[j + 1]);
                                coefficients_[j] = 1.0;
                            }

                            variables_[sz - 1] = edge;
                            coefficients_[sz - 1] = -1.0;

                            ilpSolver_.addConstraint(variables_.begin(), variables_.begin() + sz, 
                                                     coefficients_.begin(), 0, std::numeric_limits<double>::infinity());
                            ++nCycle;
                        }
                    }
                }
                std::cout<<"nCycle "<<nCycle<<"\n";
                return nCycle;
            };

            auto repairSolution = [&] (){
                for (size_t edge = 0; edge < graph_.numberOfEdges(); ++edge){
                    auto v0 = graph_.u(edge);
                    auto v1 = graph_.v(edge);

                    outputEdgeLabels[edge] = components_.areConnected(v0, v1) ? 0 : 1;
                }

                ilpSolver_.setStart(outputEdgeLabels.begin());
            };



            for (size_t i = 0; settings_.numberOfIterations == 0 || i < settings_.numberOfIterations; ++i){
                if (i != 0){
                    repairSolution();
                }
                ilpSolver_.optimize();
                if (addCycleInequalities() == 0){
                    break;
                }
            }
            repairSolution();
        }   
    private:

        void addThreeCyclesConstraintsExplicitly(const IlpSovler & ilpSolver){

        }

        void initializeIlp(IlpSovler & ilpSolver){
            std::vector<double> costs(graph_.numberOfEdges(),0.0);
            const auto & weights = objective_.weights();
            for(auto e : graph_.edges()){
                costs[e] = weights[e];
            }
            ilpSolver.initModel(graph_.numberOfEdges(), costs.data());
        }

        void addThreeCyclesConstraintsExplicitly(){
            std::array<size_t, 3> variables;
            std::array<double, 3> coefficients;
            auto threeCycles = findThreeCyclesEdges(graph_);
            std::cout<<"nThreeCycles "<<threeCycles.size()<<"\n";
            for(auto & tce : threeCycles){
                for(auto i=0; i<3; ++i){
                    variables[i] = tce[i];
                }
                for(auto i=0; i<3; ++i){
                    for(auto j=0; j<3; ++j){
                        if(i != j){
                            coefficients[i] = 1.0;
                        }
                    }
                    coefficients[i] = -1.0;
                    ilpSolver_.addConstraint(variables.begin(), variables.begin() + 3, 
                                        coefficients.begin(), 0, std::numeric_limits<double>::infinity());
                }
            }
        }

        const Objective & objective_;
        const Graph & graph_;
        IlpSovler ilpSolver_;
        Components components_;
        Settings settings_;
        std::vector<size_t> variables_;
        std::vector<double> coefficients_;
    };

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_MULTICUT_MULTICUT_ILP_HXX
