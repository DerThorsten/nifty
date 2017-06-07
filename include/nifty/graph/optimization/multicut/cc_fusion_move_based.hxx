#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"


#include "nifty/graph/optimization/multicut/fusion_move.hxx"
#include "nifty/graph/optimization/common/cc_fusion_move_based_impl.hxx"


#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_factory_base.hxx"




namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{


    template<class OBJECTIVE>
    class CcFusionMoveBased : 

    // the impl is the base 
    public nifty::graph::optimization::common::detail_cc_fusion::CcFusionMoveBasedImpl<
        OBJECTIVE, 
        MulticutBase<OBJECTIVE>,
        FusionMove<OBJECTIVE>
    >
    {
    public: 

        typedef OBJECTIVE                   ObjectiveType;
        typedef nifty::graph::optimization::common::detail_cc_fusion::CcFusionMoveBasedImpl<
            OBJECTIVE, 
            MulticutBase<OBJECTIVE>,
            FusionMove<OBJECTIVE>
        >                                   BaseType;
        typedef typename BaseType::Settings Settings;
    
        virtual ~CcFusionMoveBased(){
        }

        CcFusionMoveBased(const ObjectiveType & objective, const Settings & settings = Settings())
        :   BaseType(objective, settings){
        }

 
    };

 


} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
