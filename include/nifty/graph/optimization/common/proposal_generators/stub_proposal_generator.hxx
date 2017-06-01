
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


        struct Settings{
        };

        StubProposalGenerator(
            const ObjectiveType & objective, 
            const size_t numberOfThreads,
            const Settings & settings  = Settings()
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
            const size_t tid
        ){
            
        }
    private:
        const ObjectiveType & objective_;
        size_t numberOfThreads_;
        Settings settings_;
    }; 



} // namespace nifty::graph::optimization::common
} // namespacen ifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

