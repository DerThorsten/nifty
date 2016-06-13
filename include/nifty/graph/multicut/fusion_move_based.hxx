#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_FUSION_MOVES_HXX
#define NIFTY_GRAPH_MULTICUT_FUSION_MOVES_HXX

#include <mutex>          // std::mutex

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/multicut/multicut_base.hxx"
#include "nifty/parallel/threadpool.hxx"


namespace nifty{
namespace graph{



    template<class OBJECTIVE>
    class FusionMove{
    public:
        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;

        struct Settings{

        };

        FusionMove(const Objective & objective, const Settings & settings = Settings()){

        }

        template<class NODE_MAP_PTR_ITER, class NODE_MAP >
        void fuse(NODE_MAP_PTR_ITER toFuseBegin, NODE_MAP_PTR_ITER toFuseEnd, NODE_MAP * result ){

        }
    private:

    };




    template<class PROPPOSAL_GEN>
    class FusionMoveBased : public MulticutBase<typename PROPPOSAL_GEN::Objective >
    {
    public: 

        typedef typename PROPPOSAL_GEN::Objective Objective;
        typedef typename Objective::Graph Graph;
        typedef MulticutBase<Objective> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;

        typedef FusionMove<Objective> FusionMoveType;
        typedef PROPPOSAL_GEN ProposalGen;
        typedef typename ProposalGen::Settings ProposalGenSettings;
        typedef typename FusionMoveType::Settings FusionMoveSettings;
    private:
        typedef detail_graph::UndirectedAdjacency<int64_t,int64_t,int64_t,int64_t> NodeAdjacency;
        typedef std::set<NodeAdjacency> NodeStorage;
        typedef typename Graph:: template NodeMap<NodeStorage> NodesContainer;
        typedef std::pair<int64_t,int64_t> EdgeStorage;
        typedef typename Graph:: template EdgeMap<EdgeStorage> EdgeContainer;
        typedef typename Graph:: template EdgeMap<double> EdgeWeights;
    public:

        struct Settings{
            int verbose { 1 };
            int numberOfThreads {-1};
            size_t numberOfIterations {10};
            size_t numberOfParallelProposals {100};
            size_t fuseN{2};
            ProposalGenSettings proposalGenSettings;
            FusionMoveSettings fusionMoveSettings;
        };


        FusionMoveBased(const Objective & objective, const Settings & settings = Settings());
        ~FusionMoveBased();
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;

    private:

        void reset();
        bool stopContraction();
        void relabelEdge(const uint64_t edge,const uint64_t deadNode, const uint64_t aliveNode);

        const Objective & objective_;
        const Graph & graph_;
        Settings settings_;
        nifty::parallel::ParallelOptions parallelOptions_;

        std::vector<ProposalGen *> pgens_;
        std::vector<NodeLabels *>  proposals_;
        std::vector<NodeLabels *>  solBufferIn_;
        std::vector<NodeLabels *>  solBufferOut_;
        std::vector<FusionMoveType * > fusionMoves_;
        NodeLabels currentBest_;

    };

    template<class PROPPOSAL_GEN>
    FusionMoveBased<PROPPOSAL_GEN>::
    FusionMoveBased(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        settings_(settings),
        parallelOptions_(settings.numberOfThreads),
        pgens_(),
        proposals_(),
        currentBest_(objective.graph())
    {
        const auto nt = parallelOptions_.getActualNumThreads();
        pgens_.resize(nt);
        fusionMoves_.resize(nt);
        for(size_t i=0; i<nt; ++i){
            pgens_[i] = new ProposalGen(objective_, settings_.proposalGenSettings, i);
            fusionMoves_[i] = new FusionMoveType(objective_, settings_.fusionMoveSettings);
            solBufferIn_[i] = new NodeLabels(graph_);
            solBufferOut_[i] = new NodeLabels(graph_);
        }

        const auto np = settings_.numberOfParallelProposals*2;
        proposals_.resize(np);
        for(auto i=0; i<np; ++i){
            proposals_[i] = new NodeLabels(graph_);
        }
    }

    template<class PROPPOSAL_GEN>
    FusionMoveBased<PROPPOSAL_GEN>::
    ~FusionMoveBased(){

        for(size_t i=0; i<parallelOptions_.getActualNumThreads(); ++i){
            delete pgens_[i];
            delete fusionMoves_[i];
            delete solBufferIn_[i];
            delete solBufferOut_[i];
        }
        for(size_t i=0; i<proposals_.size(); ++i){
            delete proposals_[i];
        }
    }

    template<class PROPPOSAL_GEN>
    void FusionMoveBased<PROPPOSAL_GEN>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){

        nifty::parallel::ThreadPool threadPool(parallelOptions_);
        std::mutex mtx;


        std::vector<NodeLabels> proposals;

        for(auto iter=0; iter<settings_.numberOfIterations; ++iter){

            // generate the proposals and fuse with the current best
            nifty::parallel::parallel_foreach(threadPool,settings_.numberOfParallelProposals,
                [&](const size_t threadId, int proposalIndex){

                    // 
                    auto & pgen = *pgens_[proposalIndex];
                    auto & proposal = *solBufferIn_[threadId];
                    pgen.generate(currentBest_, proposal);

            });
        }
    }

    template< class PROPPOSAL_GEN>
    const typename FusionMoveBased<PROPPOSAL_GEN>::Objective &
    FusionMoveBased<PROPPOSAL_GEN>::
    objective()const{
        return objective_;
    }



} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_MULTICUT_FUSION_MOVES_HXX
