#pragma once


#include "nifty/tools/runtime_check.hxx"

#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/opt/multicut/multicut_factory.hxx"
#include "nifty/graph/opt/multicut/multicut_objective.hxx"






namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

   


    template<class OBJECTIVE>
    class BlockMulticut : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::WeightType WeightType;
        typedef MulticutBase<ObjectiveType> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::EdgeLabels EdgeLabels;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightsMap WeightsMap;
        typedef typename GraphType:: template EdgeMap<uint8_t> IsDirtyEdge;


        typedef MulticutFactoryBase<ObjectiveType>  McFactoryBase;


    
    public:

        struct SettingsType{
            std::shared_ptr<McFactoryBase> multicutFactory;
        };

        virtual ~BlockMulticut(){
            
        }
        BlockMulticut(const Objective & objective, const SettingsType & settings = SettingsType());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBaseType * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("BlockMulticut");
        }
        virtual void weightsChanged(){ 
        }
        virtual double currentBestEnergy() {
           return currentBestEnergy_;
        }
    private:


        const Objective & objective_;
        SettingsType settings_;
        NodeLabels * currentBest_;
        double currentBestEnergy_;
    
    };

    
    template<class OBJECTIVE>
    BlockMulticut<OBJECTIVE>::
    BlockMulticut(
        const Objective & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        settings_(settings),
        currentBest_(nullptr),
        currentBestEnergy_(std::numeric_limits<double>::infinity())
    {

    }

    template<class OBJECTIVE>
    void BlockMulticut<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBaseType * visitor
    ){  


        
        VisitorProxy visitorProxy(visitor);
        currentBest_ = &nodeLabels;
        currentBestEnergy_ = objective_.evalNodeLabels(nodeLabels);
        
        visitorProxy.begin(this);

  
        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename BlockMulticut<OBJECTIVE>::Objective &
    BlockMulticut<OBJECTIVE>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

