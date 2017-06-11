#pragma once

#include <string>
#include <random>
#include <vector>

#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_base.hxx"


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
    class RandomNodeColorProposalGenerator : 
        public ProposalGeneratorBase<OBJECTIVE>{

        
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename GraphType:: template NodeMap<uint64_t>  ProposalType;
        typedef typename GraphType:: template NodeMap<bool>  IsUsed;    
            
        struct SettingsType{
            size_t numberOfColors{size_t(2)};
        };

        RandomNodeColorProposalGenerator(
            const ObjectiveType & objective, 
            const size_t numberOfThreads,
            const SettingsType & settings  = SettingsType()
        )
        :   objective_(objective),
            numberOfThreads_(numberOfThreads),
            settings_(settings),
            gens_(numberOfThreads_),
            colorDist_(0,settings.numberOfColors-1)
        {
            // use thread index as seed
            for(auto i=0; i<numberOfThreads_; ++i){
                gens_[i] = std::mt19937(i);
            }
            this->reset();
        }
        void reset(){

        }
        virtual ~RandomNodeColorProposalGenerator(){}

        virtual void generateProposal(
            const ProposalType & currentBest, 
            ProposalType & proposal, 
            const size_t tid
        ){  

            const auto & graph = objective_.graph();
            auto & gen = gens_[tid];
   

            // set to unused
            // and copy current best
            graph.forEachNode([&](const uint64_t node){
                proposal[node] = colorDist_(gen);
            });
        }
    private:
        const ObjectiveType & objective_;
        size_t numberOfThreads_;
        SettingsType settings_;
        std::vector<std::mt19937> gens_;
        std::uniform_int_distribution<>  colorDist_;
    }; 


} // namespace nifty::graph::optimization::common
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
