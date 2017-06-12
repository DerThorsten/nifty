#pragma once

#include <string>
#include <random>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/edge_weighted_watersheds.hxx"


namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

    template<class OBJECTIVE>
    class WatershedProposals{
    public:
        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef MulticutBase<Objective> Base;
        typedef MulticutGreedyAdditive<Objective> Solver;
        typedef typename Solver::SettingsType SolverSettings;
        typedef typename Base::NodeLabelsType NodeLabelsType;
        typedef typename Graph:: template EdgeMap<double>  EdgeWeights;

        struct SettingsType{
            double sigma{1.0};
            double seedFraction{0.1};
        };

        static std::string name(){
            return std::string("WatershedProposals");
        }

        WatershedProposals(
            const Objective & objective, 
            const SettingsType  & settings,
            const size_t threadIndex
        )
        :   objective_(objective),
            graph_(objective.graph()),
            weights_(objective.graph()),
            seeds_(objective.graph()),
            negativeEdges_(),
            settings_(settings),
            threadIndex_(threadIndex),
            proposalNumber_(0),
            gen_(threadIndex),
            dist_(0.0, settings.sigma),
            intDist_()
        {
            this->reset();
        }

        ~WatershedProposals(){
            
        }

        void generate( const NodeLabelsType & currentBest, NodeLabelsType & proposal){
            if(negativeEdges_.empty()){
                proposal = currentBest;
            }
            else{
                size_t nSeeds = settings_.seedFraction <=1.0 ? 
                    size_t(float(graph_.numberOfNodes())*settings_.seedFraction+0.5f) :
                    size_t(settings_.seedFraction + 0.5);

                nSeeds = std::max(size_t(1),nSeeds);
                nSeeds = std::min(size_t(negativeEdges_.size()-1), nSeeds);


                // get the seeds
                for(const auto node : graph_.nodes())
                    seeds_[node] = 0;


                for(size_t i=0; i <  (nSeeds == 1 ? 1 : nSeeds/2); ++i){
                    const auto randIndex = intDist_(gen_);
                    const auto edge  = negativeEdges_[randIndex];

                    const auto v0 = graph_.u(edge);
                    const auto v1 = graph_.v(edge);

                    seeds_[v0] = (2*i)+1;
                    seeds_[v1] = (2*i+1)+1;
                }


                // randomize the weights
                const auto & weights = objective_.weights();
                for(auto edge: graph_.edges()){
                    weights_[edge] = -1.0*weights[edge] + dist_(gen_);
                }
                edgeWeightedWatershedsSegmentation(graph_, weights_, seeds_, proposal);
                ++proposalNumber_;
            }
        }
        void reset(){
            proposalNumber_ = 0;
            negativeEdges_.resize(0);
            const auto & weights = objective_.weights();
            for(auto edge: graph_.edges()){
                if(weights[edge]<0.0){
                    negativeEdges_.push_back(edge);
                }
            }
            if(!negativeEdges_.empty())
                intDist_ = std::uniform_int_distribution<> (0, negativeEdges_.size()-1);
            else{
                // fallback to not crash, but meaningless since there are no negative edges
                intDist_ = std::uniform_int_distribution<> (0, 1);
            }
        }

    private:

        const Objective & objective_;
        const Graph graph_;
        EdgeWeights weights_;
        NodeLabelsType seeds_;
        std::vector<uint64_t> negativeEdges_;
        SettingsType settings_;
        size_t threadIndex_;
        size_t proposalNumber_;
        std::mt19937 gen_;
        std::normal_distribution<> dist_;
        std::uniform_int_distribution<>  intDist_;
    };



} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

