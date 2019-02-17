
#pragma once

#include <string>
#include <random>
#include <vector>

#include "nifty/graph/opt/common/proposal_generators/proposal_generator_base.hxx"


namespace nifty{
namespace graph{
namespace opt{
namespace common{


    /**
     * @brief Stub for proposal generator
     * @details good staring point for a proposal generator
     * 
     * @tparam OBJECTIVE [description]
     */
    template<class OBJECTIVE>
    class StubProposalGenerator : 
        public ProposalGeneratorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename GraphType:: template NodeMap<uint64_t>  ProposalType;


        struct SettingsType{
        };

        StubProposalGenerator(
            const ObjectiveType & objective, 
            const std::size_t numberOfThreads,
            const SettingsType & settings  = SettingsType()
        )
        :   objective_(objective),
            numberOfThreads_(numberOfThreads),
            settings_(settings)
        {

            
        }


        virtual ~StubProposalGenerator(){}

        virtual void generateProposal(
            const ProposalType & currentBest, 
            ProposalType & proposal, 
            const std::size_t tid
        ){
            
        }
    private:
        const ObjectiveType & objective_;
        std::size_t numberOfThreads_;
        SettingsType settings_;
    }; 



} // namespace nifty::graph::opt::common
} // namespacen ifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

