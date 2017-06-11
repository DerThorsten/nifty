
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
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/graph/optimization/lifted_multicut/fusion_move.hxx"

#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_factory_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_factory.hxx"
#include "nifty/graph/optimization/lifted_multicut/proposal_generators/watershed_proposal_generator.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{








    /**
     * @brief      Class for fusion move based inference for the lifted multicut objective
     *             An implementation of \cite beier_efficient_2016.
     *             
     *          
     * @tparam     OBJECTIVE  { description }
     */
    template<class OBJECTIVE>
    class FusionMoveBased : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> BaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;

        typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;
        typedef ProposalGeneratorFactoryBase<ObjectiveType> ProposalGeneratorFactoryBaseType;

        typedef WatershedProposalGenerator<ObjectiveType> DefaultProposalGeneratorType;
        typedef ProposalGeneratorFactory<DefaultProposalGeneratorType> DefaultProposalGeneratorFactoryType;

    private:
        typedef FusionMove<ObjectiveType> FusionMoveType;

    

    public:

        struct Settings{
            std::shared_ptr<ProposalGeneratorFactoryBaseType> proposalGeneratorFactory;
            size_t numberOfIterations{1000};
            size_t stopIfNoImprovement{10};
            int numberOfThreads{1};
        };



        virtual ~FusionMoveBased();
        FusionMoveBased(const ObjectiveType & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;





 


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("FusionMoveBased");
        }


    private:
        virtual void optimizeSingleThread(VisitorProxyType & visitorProxy);
        virtual void optimizeMultiThread(VisitorProxyType & visitorProxy);


        const ObjectiveType & objective_;
        Settings settings_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        NodeLabelsType * currentBest_;
        double currentBestEnergy_;

        nifty::parallel::ParallelOptions parallelOptions_;
        nifty::parallel::ThreadPool threadPool_;
        ProposalGeneratorBaseType * proposalGenerator_;

        std::vector<FusionMoveType *> fusionMoves_;
    };

    
    template<class OBJECTIVE>
    FusionMoveBased<OBJECTIVE>::
    FusionMoveBased(
        const ObjectiveType & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        liftedGraph_(objective.liftedGraph()),
        currentBest_(nullptr),
        currentBestEnergy_(0),
        parallelOptions_(settings.numberOfThreads),
        threadPool_(parallelOptions_),
        fusionMoves_()
    {
        if(!bool(settings_.proposalGeneratorFactory)){
            auto pgenSettings  = typename DefaultProposalGeneratorType::Settings();
            settings_.proposalGeneratorFactory =  std::make_shared<DefaultProposalGeneratorFactoryType>(pgenSettings);
        }

        const auto numberOfThreads = parallelOptions_.getActualNumThreads();

        // generate proposal generators
        proposalGenerator_ = settings_.proposalGeneratorFactory->create(objective_, numberOfThreads);


        // generate fusion moves
        fusionMoves_.resize(numberOfThreads);


        parallel::parallel_foreach(threadPool_, numberOfThreads, [&](const int tid, const int i){
            fusionMoves_[i] = new FusionMoveType(objective_);
        });


    }


    template<class OBJECTIVE>
    FusionMoveBased<OBJECTIVE>::
    ~FusionMoveBased(){

        // delete proposal generator
        delete proposalGenerator_;

        const auto numberOfThreads = parallelOptions_.getActualNumThreads();
        parallel::parallel_foreach(threadPool_,numberOfThreads,
        [&](const int tid, const int i){
            delete fusionMoves_[i];
        });
    }

    template<class OBJECTIVE>
    void FusionMoveBased<OBJECTIVE>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){
        
        // set starting point as current best
        currentBest_ = &nodeLabels;
        currentBestEnergy_ = objective_.evalNodeLabels(*currentBest_);



        VisitorProxyType visitorProxy(visitor);
        visitorProxy.begin(this);



        if(parallelOptions_.getActualNumThreads() == 1){
            this->optimizeSingleThread(visitorProxy);
        }
        else{
            this->optimizeMultiThread(visitorProxy);
        }

        visitorProxy.end(this);     
    }

    template<class OBJECTIVE>
    void FusionMoveBased<OBJECTIVE>::
    optimizeSingleThread(
        VisitorProxyType & visitorProxy
    ){
        NodeLabelsType proposal(graph_);
        auto iterWithoutImprovement = 0;

        for(auto iteration=0; iteration<settings_.numberOfIterations; ++iteration){
           
            // generate a proposal
            proposalGenerator_->generateProposal(*currentBest_, proposal, 0);

            // eval energy of proposal
            //std::cout<<"E "<<objective_.evalNodeLabels(proposal)<<"\n";

            // accept the first proposal as current best
            // if starting point was trivial (one connected comp.)
            if(currentBestEnergy_ >=-0.000000001 && iteration==0){
                graph_.forEachNode([&](const uint64_t node){
                    (*currentBest_)[node] = proposal[node];    
                });
                currentBestEnergy_  = objective_.evalNodeLabels(*currentBest_);
            }
            else{
                NodeLabelsType res(graph_);
                auto & fm = *(fusionMoves_[0]);
                fm.fuse( {&proposal, currentBest_}, &res);
                auto eFuse = objective_.evalNodeLabels(res);
                if(eFuse<currentBestEnergy_){
                    currentBestEnergy_ = eFuse;
                    graph_.forEachNode([&](const uint64_t node){
                        (*currentBest_)[node] = res[node];    
                    });
                    iterWithoutImprovement = 0;
                }   
                else{
                    ++iterWithoutImprovement;
                    if(iterWithoutImprovement >= settings_.stopIfNoImprovement){
                        break;
                    }
                }
            }

            visitorProxy.visit(this);
        }
    }

    template<class OBJECTIVE>
    void FusionMoveBased<OBJECTIVE>::
    optimizeMultiThread(
        VisitorProxyType & visitorProxy
    ){
        NIFTY_CHECK(false, "currently only single thread is implemented");
    }

    template<class OBJECTIVE>
    const typename FusionMoveBased<OBJECTIVE>::ObjectiveType &
    FusionMoveBased<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 


    
} // lifted_multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

