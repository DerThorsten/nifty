#pragma once

#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <stack>

#include "nifty/tools/changable_priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/opt/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/components.hxx"

#include "nifty/graph/opt/common/solver_factory_base.hxx"
#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/opt/multicut/multicut_objective.hxx"



namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{










    template<class OBJECTIVE>
    class LiftedMulticutViaMulticut : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> BaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxyType VisitorProxy;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;

    private:

        typedef nifty::graph::opt::multicut::MulticutBase<ObjectiveType> McBaseType;
        typedef nifty::graph::opt::common::SolverFactoryBase<McBaseType>  McFactoryBase;

    public:


        struct SettingsType {
            std::shared_ptr<McFactoryBase> multicutFactory;
        };

        



        virtual ~LiftedMulticutViaMulticut(){}
        LiftedMulticutViaMulticut(const ObjectiveType & objective, const SettingsType & settings = SettingsType());
        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;





 


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("LiftedMulticutViaMulticut");
        }


    private:




        const ObjectiveType & objective_;
        SettingsType settings_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        NodeLabelsType * currentBest_;


        // multicut objective
        nifty::graph::opt::multicut::MulticutObjective<LiftedGraphType, double> multicutObjective_;

    };

    
    template<class OBJECTIVE>
    LiftedMulticutViaMulticut<OBJECTIVE>::
    LiftedMulticutViaMulticut(
        const ObjectiveType & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        liftedGraph_(objective.liftedGraph()),
        currentBest_(nullptr),
        multicutObjective_(objective.liftedGraph())
    {
       for(auto e : liftedGraph_.edges()){
            multicutObjective_.weights()[e] = objective_.weights()[e]; 
       }
    }

    template<class OBJECTIVE>
    void LiftedMulticutViaMulticut<OBJECTIVE>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){
        
        currentBest_ = &nodeLabels;

        VisitorProxy visitorProxy(visitor);
        visitorProxy.begin(this);     

        // run multicut solver...

        visitorProxy.end(this);     
    }

    template<class OBJECTIVE>
    const typename LiftedMulticutViaMulticut<OBJECTIVE>::ObjectiveType &
    LiftedMulticutViaMulticut<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 


    
} // lifted_multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

