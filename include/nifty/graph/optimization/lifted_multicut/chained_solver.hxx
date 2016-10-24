#pragma once


#include <memory>


#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_factory.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_visitor_base.hxx"


namespace nifty{
namespace graph{
namespace lifted_multicut{




    template<class OBJECTIVE>
    class ChainedSolver : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef typename Objective::GraphType GraphType;
        typedef typename Objective::LiftedGraphType LiftedGraphType;
        typedef LiftedMulticutBase<OBJECTIVE> BaseType;
        typedef typename BaseType::VisitorBase VisitorBase;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::EdgeLabels EdgeLabels;
        typedef typename BaseType::NodeLabels NodeLabels;


        typedef LiftedMulticutFactoryBase<Objective> FactoryBase;

    public:

        struct Settings{
            std::shared_ptr<FactoryBase> factoryA;
            std::shared_ptr<FactoryBase> factoryB;
        };


        virtual ~ChainedSolver(){
            delete solverA_;
            delete solverB_;
        }
        ChainedSolver(const Objective & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;

        void reset();


        //virtual void weightsChanged(){
        //    this->reset();
        //}

        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("ChainedSolver");
        }


    private:


        const Objective & objective_;
        const GraphType & graph_;
        NodeLabels * currentBest_;


        BaseType * solverA_;
        BaseType * solverB_;

    };

    
    template<class OBJECTIVE>
    ChainedSolver<OBJECTIVE>::
    ChainedSolver(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        currentBest_(nullptr),
        solverA_(settings.factoryA->createRawPtr(objective)),
        solverB_(settings.factoryB->createRawPtr(objective))
    {

    }

    template<class OBJECTIVE>
    void ChainedSolver<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        currentBest_ = &nodeLabels;

        VisitorProxy visitorProxy(visitor);
        visitorProxy.begin(this);
        
        solverA_->optimize(nodeLabels, nullptr);
        visitorProxy.visit(this);


        solverB_->optimize(nodeLabels, nullptr);
        visitorProxy.visit(this);


        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename ChainedSolver<OBJECTIVE>::Objective &
    ChainedSolver<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 
    template<class OBJECTIVE>
    void ChainedSolver<OBJECTIVE>::
    reset(
    ){
        solverA_->reset();
        solverB_->reset();
    }

    
} // lifted_multicut
} // namespace nifty::graph
} // namespace nifty
