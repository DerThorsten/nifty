
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

