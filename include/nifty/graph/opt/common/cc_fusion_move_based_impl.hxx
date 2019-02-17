#pragma once

#include <mutex>          // std::mutex


#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <stack>

#include "nifty/tools/changable_priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/opt/mincut/mincut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/parallel/threadpool.hxx"

#include "nifty/graph/opt/common/proposal_generators/proposal_generator_base.hxx"
#include "nifty/graph/opt/common/proposal_generators/proposal_generator_factory_base.hxx"


namespace nifty{
namespace graph{
namespace opt{
namespace common{
namespace detail_cc_fusion{






    // mincut and multicut and lifted multicut use the same basic fusion move solver
    template<
        class OBJECTIVE, 
        class SOLVER_BASE, 
        class FUSION_MOVE
    >
    class CcFusionMoveBasedImpl : public SOLVER_BASE
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef SOLVER_BASE BaseType;
        typedef typename ObjectiveType::GraphType GraphType;
     
        
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;



        typedef nifty::graph::opt::common::ProposalGeneratorBase<ObjectiveType>        ProposalGeneratorBaseType;
        typedef nifty::graph::opt::common::ProposalGeneratorFactoryBase<ObjectiveType> ProposalGeneratorFactoryBaseType;


    private:
        typedef FUSION_MOVE FusionMoveType;
        typedef typename  FusionMoveType::SettingsType FusionMoveSettingsType;

    

    public:

        struct SettingsType{
            std::shared_ptr<ProposalGeneratorFactoryBaseType> proposalGeneratorFactory;
            std::size_t numberOfIterations{1000};
            std::size_t stopIfNoImprovement{10};
            int numberOfThreads{1};
            int numberOfParallelProposals{-1};
            FusionMoveSettingsType fusionMoveSettings;
        };



        virtual ~CcFusionMoveBasedImpl();
        CcFusionMoveBasedImpl(const ObjectiveType & objective, const SettingsType & settings = SettingsType());
        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("CcFusionMoveBasedImpl");
        }


    private:
        virtual void optimizeSingleThread(VisitorProxyType & visitorProxy);
        virtual void optimizeMultiThread(VisitorProxyType & visitorProxy);


        const ObjectiveType & objective_;
        SettingsType settings_;
        const GraphType & graph_;
        NodeLabelsType * currentBest_;
        double currentBestEnergy_;

        nifty::parallel::ParallelOptions parallelOptions_;
        nifty::parallel::ThreadPool threadPool_;
        ProposalGeneratorBaseType * proposalGenerator_;

        std::vector<FusionMoveType *> fusionMoves_;
        std::vector<NodeLabelsType *>     solBufferIn_;
        std::vector<NodeLabelsType *>     solBufferOut_;
    };

    
    template<class OBJECTIVE, class SOLVER_BASE, class FUSION_MOVE>
    CcFusionMoveBasedImpl<OBJECTIVE, SOLVER_BASE, FUSION_MOVE>::
    CcFusionMoveBasedImpl(
        const ObjectiveType & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        currentBest_(nullptr),
        currentBestEnergy_(0),
        parallelOptions_(settings.numberOfThreads),
        threadPool_(parallelOptions_),
        fusionMoves_()
    {
        if(!bool(settings_.proposalGeneratorFactory)){
            throw std::runtime_error("proposalGeneratorFactory shall not be empty!");
            // auto pgenSettings  = typename DefaultProposalGeneratorType::SettingsType();
            // settings_.proposalGeneratorFactory =  std::make_shared<DefaultProposalGeneratorFactoryType>(pgenSettings);
        }


        const auto numberOfThreads = parallelOptions_.getActualNumThreads();

        if(settings_.numberOfParallelProposals<0){
            settings_.numberOfParallelProposals = numberOfThreads;
        }

        // generate proposal generators
        proposalGenerator_ = settings_.proposalGeneratorFactory->create(objective_, numberOfThreads);


        // generate fusion moves
        fusionMoves_.resize(numberOfThreads);
        solBufferIn_.resize(numberOfThreads);

        parallel::parallel_foreach(threadPool_, numberOfThreads, [&](const int tid, const int i){
            fusionMoves_[i] = new FusionMoveType(objective_, settings_.fusionMoveSettings);
            solBufferIn_[i] = new NodeLabelsType(graph_);
        });


    }


    template<class OBJECTIVE, class SOLVER_BASE, class FUSION_MOVE>
    CcFusionMoveBasedImpl<OBJECTIVE, SOLVER_BASE, FUSION_MOVE>::
    ~CcFusionMoveBasedImpl(){

        // delete proposal generator
        delete proposalGenerator_;

        const auto numberOfThreads = parallelOptions_.getActualNumThreads();
        parallel::parallel_foreach(threadPool_,numberOfThreads,
        [&](const int tid, const int i){
            delete fusionMoves_[i];
            delete solBufferIn_[i];
        });
    }

    template<class OBJECTIVE, class SOLVER_BASE, class FUSION_MOVE>
    void CcFusionMoveBasedImpl<OBJECTIVE, SOLVER_BASE, FUSION_MOVE>::
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

    template<class OBJECTIVE, class SOLVER_BASE, class FUSION_MOVE>
    void CcFusionMoveBasedImpl<OBJECTIVE, SOLVER_BASE, FUSION_MOVE>::
    optimizeSingleThread(
        VisitorProxyType & visitorProxy
    ){
        NodeLabelsType proposal(graph_);
        auto iterWithoutImprovement = 0;

        for(auto iteration=0; iteration<settings_.numberOfIterations; ++iteration){
           
            // generate a proposal
            proposalGenerator_->generateProposal(*currentBest_, proposal, 0);

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

            if(!visitorProxy.visit(this))
                break;
        }
    }

    template<class OBJECTIVE, class SOLVER_BASE, class FUSION_MOVE>
    void CcFusionMoveBasedImpl<OBJECTIVE, SOLVER_BASE, FUSION_MOVE>::
    optimizeMultiThread(
        VisitorProxyType & visitorProxy
    ){
        // NIFTY_CHECK(false, "currently only single thread is implemented");


        std::mutex mtxCurrentBest;
        std::mutex mtxProposals;
        std::vector<NodeLabelsType> proposals;


        auto & currentBest = *currentBest_;

        auto nWithoutImprovment = 0;
        for(auto iteration=0; iteration<settings_.numberOfIterations; ++iteration){

            const auto oldBestEnergy = currentBestEnergy_;
            proposals.clear();



            visitorProxy.printLog(nifty::logging::LogLevel::INFO, 
                std::string("Generating Proposals (and fuse with current best)"));


            nifty::parallel::parallel_foreach(threadPool_,
                settings_.numberOfParallelProposals,
                [&](const std::size_t threadId, int proposalIndex){

                    
                    NodeLabelsType currentBestBuffer(graph_);

                    // buffer current best
                    mtxCurrentBest.lock();
                    for(const auto node : graph_.nodes()){
                        currentBestBuffer[node] = currentBest[node];
                    }
                    mtxCurrentBest.unlock();

                    // generate a proposal
                    auto & proposal = *solBufferIn_[threadId];
                    proposalGenerator_->generateProposal(currentBestBuffer, proposal, threadId);
                        

                    // evaluate the energy of the proposal
                    const auto eProposal = objective_.evalNodeLabels(proposal);


                    if(currentBestEnergy_ >= -0.0000001 && iteration == 0){

                        // just accept this as current best
                        mtxCurrentBest.lock();
                        for(const auto node : graph_.nodes()){
                            currentBest[node] = proposal[node];
                        }
                        currentBestEnergy_ = eProposal;
                        mtxCurrentBest.unlock();

                    }
                    else{
                        // fuse with the current best
                        NodeLabelsType res(graph_);
                        auto & fm = *(fusionMoves_[threadId]);
                        fm.fuse( {&proposal, &currentBestBuffer}, &res);
                        const auto eFused = objective_.evalNodeLabels(res);

                        if(eFused < currentBestEnergy_){
                            mtxCurrentBest.lock();
                            for(const auto node : graph_.nodes()){
                                currentBest[node] = res[node];
                            }
                            currentBestEnergy_ = eFused;
                            mtxCurrentBest.unlock();
                        }
                        else if(eFused + 0.0000001 >= currentBestEnergy_){
                            // the same...

                        }
                        else{
                            // remember
                            mtxProposals.lock();
                            proposals.push_back(res);
                            mtxProposals.unlock();
                        }
                    }
                }
            );

            
            if(proposals.size() >= 2){

                visitorProxy.printLog(nifty::logging::LogLevel::INFO, 
                    std::string("Fuse proposals #")+std::to_string(proposals.size()));

                std::vector<const NodeLabelsType*> toFuse;
                for(const auto & p : proposals){
                    toFuse.push_back(&p);
                }  

                auto & fm = *(fusionMoves_[0]);
                fm.fuse( toFuse, &currentBest);
                currentBestEnergy_ = objective_.evalNodeLabels(currentBest);
            }

            if(currentBestEnergy_ < oldBestEnergy){
                if(!visitorProxy.visit(this)){
                    break;
                }
                nWithoutImprovment = 0;
            }
            else{
                ++nWithoutImprovment;
            }

            if(nWithoutImprovment >= settings_.stopIfNoImprovement){
                break;
            }



        }






    }

    template<class OBJECTIVE, class SOLVER_BASE, class FUSION_MOVE>
    const typename CcFusionMoveBasedImpl<OBJECTIVE, SOLVER_BASE, FUSION_MOVE>::ObjectiveType &
    CcFusionMoveBasedImpl<OBJECTIVE, SOLVER_BASE, FUSION_MOVE>::
    objective()const{
        return objective_;
    }

 

} // namespace nifty::graph::opt::common::detail_cc_fusion
} // namespace nifty::graph::opt::common
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

