#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_PROPOSALS_GREEDY_ADDITIVE_PROPOSALS_HXX
#define NIFTY_GRAPH_MULTICUT_PROPOSALS_GREEDY_ADDITIVE_PROPOSALS_HXX

#include <string>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/multicut/multicut_base.hxx"
#include "nifty/graph/multicut/multicut_greedy_additive.hxx"


namespace nifty{
namespace graph{


    template<class OBJECTIVE>
    class GreedyAdditiveProposals{
    public:
        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef MulticutBase<Objective> Base;
        typedef MulticutGreedyAdditive<Objective> Solver;
        typedef typename Solver::Settings SolverSettings;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;

        struct Settings{
            double sigma{1.0};
            double weightStopCond{0.0};
            double nodeNumStopCond{-1.0};
        };


        static std::string name(){
            return std::string("GreedyAdditiveProposals");
        }

        GreedyAdditiveProposals(
            const Objective & objective, 
            const Settings  & settings,
            const size_t threadIndex
        )
        :   objective_(objective),
            graph_(objective.graph()),
            settings_(settings),
            threadIndex_(threadIndex),
            proposalNumber_(0),
            solver_()

        {
            SolverSettings solverSettings;
            solverSettings.verbose = 0;
            solverSettings.addNoise = true;
            solverSettings.sigma = settings_.sigma;
            solverSettings.weightStopCond  = settings_.weightStopCond;
            solverSettings.nodeNumStopCond  = settings_.nodeNumStopCond;
            solverSettings.seed = threadIndex_;
            solver_ = new Solver(objective_, solverSettings);
        }

        ~GreedyAdditiveProposals(){
            delete solver_;
        }

        void generate( const NodeLabels & currentBest, NodeLabels & proposal){
            if(proposalNumber_ != 0){
                solver_->reset();
            }
            // maybe refactor the warm start 
            for(const auto node : graph_.nodes()){
                proposal[node] = currentBest[node];
            }

            MulticutEmptyVisitor<Objective> visitor;

            solver_->optimize(proposal, &visitor);
            ++proposalNumber_;
        }

        void reset(){
            proposalNumber_ = 0;
            solver_->reset();
        }

    private:

        const Objective & objective_;
        const Graph graph_;
        Settings settings_;
        size_t threadIndex_;
        size_t proposalNumber_;

        Solver * solver_;

    };





} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_MULTICUT_PROPOSALS_GREEDY_ADDITIVE_PROPOSALS_HXX
