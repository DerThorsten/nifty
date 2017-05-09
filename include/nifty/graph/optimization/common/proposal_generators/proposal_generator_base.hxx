
#pragma once

namespace nifty{
namespace graph{
namespace optimization{
namespace common{



    template<class NODE_LABELS>
    class ProposalGeneratorBase{
    public:
        typedef NODE_LABELS NodeLabels;
        virtual ~ProposalGeneratorBase(){}
        virtual void generateProposal( const NodeLabels & currentBest,NodeLabels & labels, const size_t tid) = 0;
    }; 

    
    
    
}
}
}
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_BASE_HXX2