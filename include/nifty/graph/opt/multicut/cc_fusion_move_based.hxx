#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"


#include "nifty/graph/opt/multicut/fusion_move.hxx"
#include "nifty/graph/opt/common/cc_fusion_move_based_impl.hxx"


#include "nifty/graph/opt/common/proposal_generators/proposal_generator_factory_base.hxx"




namespace nifty{
namespace graph{
namespace opt{
namespace multicut{


    template<class OBJECTIVE>
    class CcFusionMoveBased : 

    // the impl is the base 
    public nifty::graph::opt::common::detail_cc_fusion::CcFusionMoveBasedImpl<
        OBJECTIVE, 
        MulticutBase<OBJECTIVE>,
        FusionMove<OBJECTIVE>
    >
    {
    public: 

        typedef OBJECTIVE                   ObjectiveType;
        typedef nifty::graph::opt::common::detail_cc_fusion::CcFusionMoveBasedImpl<
            OBJECTIVE, 
            MulticutBase<OBJECTIVE>,
            FusionMove<OBJECTIVE>
        >                                   BaseType;
        typedef typename BaseType::SettingsType SettingsType;
    
        virtual ~CcFusionMoveBased(){
        }

        CcFusionMoveBased(const ObjectiveType & objective, const SettingsType & settings = SettingsType())
        :   BaseType(objective, settings){
        }

 
    };

 


} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty
