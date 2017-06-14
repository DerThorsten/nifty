
#pragma once

namespace nifty{
namespace graph{
namespace opt{
namespace common{



    template<class OBJECTIVE>
    class ProposalGeneratorBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType:: template NodeMap<uint64_t> ProposalType;
        virtual ~ProposalGeneratorBase(){}
        virtual void generateProposal( const ProposalType & currentBest,ProposalType & labels, const size_t tid) = 0;
    }; 

    
    
    

} // namespace nifty::graph::opt::common
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty


