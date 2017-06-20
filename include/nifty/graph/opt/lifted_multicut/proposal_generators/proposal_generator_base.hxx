
#pragma once


#include <memory>
#include "nifty/graph/opt/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{



    template<class OBJECTIVE>
    class ProposalGeneratorBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> LiftedMulticutBaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        typedef typename LiftedMulticutBaseType::NodeLabelsType NodeLabelsType;
    
        virtual ~ProposalGeneratorBase(){}

        virtual void generateProposal( const NodeLabelsType & currentBest,NodeLabelsType & labels, const size_t tid) = 0;

    private:
    }; 

    
    
    

}
}
}
}

