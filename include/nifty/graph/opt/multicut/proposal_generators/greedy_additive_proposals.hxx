#pragma once

#include <string>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/opt/multicut/multicut_greedy_additive.hxx"


namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    template<class OBJECTIVE>
    class GreedyAdditiveProposals{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef MulticutBase<ObjectiveType> Base;
        typedef MulticutGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SolverSettings;
        typedef typename Base::NodeLabelsType NodeLabelsType;

        struct SettingsType{
            double sigma{1.0};
            double weightStopCond{0.0};
            double nodeNumStopCond{-1.0};
        };


        static std::string name(){
            return std::string("GreedyAdditiveProposals");
        }

        GreedyAdditiveProposals(
            const ObjectiveType & objective, 
            const SettingsType  & settings,
            const std::size_t threadIndex
        )
        :   objective_(objective),
            graph_(objective.graph()),
            settings_(settings),
            threadIndex_(threadIndex),
            proposalNumber_(0),
            solver_()

        {
            SolverSettings solverSettings;
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

        void generate( const NodeLabelsType & currentBest, NodeLabelsType & proposal){
            if(proposalNumber_ != 0){
                solver_->reset();
            }
            // maybe refactor the warm start 
            for(const auto node : graph_.nodes()){
                proposal[node] = currentBest[node];
            }

            MulticutEmptyVisitor<ObjectiveType> visitor;

            solver_->optimize(proposal, &visitor);
            ++proposalNumber_;
        }

        void reset(){
            proposalNumber_ = 0;
            solver_->reset();
        }

    private:

        const ObjectiveType & objective_;
        const GraphType graph_;
        SettingsType settings_;
        std::size_t threadIndex_;
        std::size_t proposalNumber_;

        Solver * solver_;

    };



} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

