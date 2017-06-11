
#pragma once

#include <string>
#include <random>
#include <vector>

#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_base.hxx"
#include "nifty/graph/edge_weighted_watersheds.hxx"


namespace nifty{
namespace graph{
namespace optimization{
namespace common{


    /**
     * @brief      Watershed proposal generator for lifted_multicut::FusionMoveBased
     *
     * @tparam     OBJECTIVE  { description }
     */
    template<class OBJECTIVE>
    class InterfaceFlipperProposalGenerator : 
        public ProposalGeneratorBase<OBJECTIVE>{

        
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename GraphType:: template NodeMap<uint64_t>  ProposalType;
        typedef typename GraphType:: template NodeMap<bool>  IsUsed;    
            
        struct SettingsType{
        };

        InterfaceFlipperProposalGenerator(
            const ObjectiveType & objective, 
            const size_t numberOfThreads,
            const SettingsType & settings  = SettingsType()
        )
        :   objective_(objective),
            numberOfThreads_(numberOfThreads),
            settings_(settings),
            gens_(numberOfThreads_),
            isUsedVec_(numberOfThreads),
            binaryDist_(0,1)
        {
            // use thread index as seed
            for(auto i=0; i<numberOfThreads_; ++i){
                gens_[i] = std::mt19937(i);
                isUsedVec_[i] = std::unique_ptr<IsUsed>(new IsUsed(objective.graph()));
            }

            this->reset();
        }

        void reset(){

        }

        virtual ~InterfaceFlipperProposalGenerator(){}

        virtual void generateProposal(
            const ProposalType & currentBest, 
            ProposalType & proposal, 
            const size_t tid
        ){  

            const auto & graph = objective_.graph();
            auto & gen = gens_[tid];
            auto & isUsed = *(isUsedVec_[tid].get());

            // set to unused
            // and copy current best
            graph.forEachNode([&](const uint64_t node){
                isUsed[node]=false;
                proposal[node] = currentBest[node];
            });


            graph.forEachEdge([&](const uint64_t edge){
                const auto uv = graph.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;
                if(!isUsed[u] && !isUsed[v]){
                    const auto lu = proposal[u];
                    const auto lv = proposal[v];
                    if(lu != lv){
                        if(binaryDist_(gen)){
                            proposal[u] = lv;
                        }
                        else{
                            proposal[v] = lu;
                        }
                        isUsed[u] = true;
                        isUsed[v] = true;
                    }
                }
            });

        }
    private:
        const ObjectiveType & objective_;
        size_t numberOfThreads_;
        SettingsType settings_;
        std::vector<std::mt19937> gens_;
        std::vector< std::unique_ptr<IsUsed> > isUsedVec_;
        std::uniform_int_distribution<>  binaryDist_;
    }; 


} // namespace nifty::graph::optimization::common
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
