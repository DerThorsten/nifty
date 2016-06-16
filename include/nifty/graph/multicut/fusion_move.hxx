#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_FUSION_MOVE_HXX
#define NIFTY_GRAPH_MULTICUT_FUSION_MOVE_HXX

#include <mutex>          // std::mutex
#include <memory>

#include "nifty/graph/multicut/multicut_greedy_additive.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/multicut/multicut_base.hxx"
#include "nifty/graph/multicut/multicut_factory.hxx"
#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"

namespace nifty{
namespace graph{



    template<class OBJECTIVE>
    class FusionMove{
    public:
        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;
        typedef MulticutFactoryBase<Objective> McFactoryBase;

        typedef UndirectedGraph<> FmGraph;
        typedef MulticutObjective<FmGraph, double> FmObjective;
        typedef MulticutBase<FmObjective> FmMcBase;
        typedef MulticutEmptyVisitor<FmObjective> FmEmptyVisitor;
        typedef typename  FmMcBase::NodeLabels FmNodeLabels;

        struct Settings{
            std::shared_ptr<McFactoryBase> mcFactory;
        };

        FusionMove(const Objective & objective, const Settings & settings = Settings())
        :   objective_(objective),
            graph_(objective.graph()),
            settings_(settings),
            ufd_(objective.graph().maxNodeId()+1),
            nodeToDense_(objective.graph())
        {
            if(!bool(settings_.mcFactory)){
                typedef MulticutGreedyAdditive<FmObjective> FmSolver;
                typedef MulticutFactory<FmSolver> FmFactory;
                settings_.mcFactory = std::make_shared<FmFactory>();
            }
        }

        template<class NODE_MAP>
        void fuse(
            std::initializer_list<NODE_MAP *> proposals,
            NODE_MAP * result
        ){
            std::vector<NODE_MAP *> p(proposals);
            fuse(p, result);
        }


        template<class NODE_MAP >
        void fuse(
            const std::vector<NODE_MAP *> & proposals,
            NODE_MAP * result 
        ){
            // reset the ufd
            ufd_.reset();

            // build the connected components
            for(const auto edge : graph_.edges()){
                // merge two nodes iff all proposals agree to merge
                bool merge = true;
                const auto uv = graph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;

                for(auto p=0; p<proposals.size(); ++p){

                    if(proposals[p]->operator[](u) != proposals[p]->operator[](v) ){
                        merge = false;
                        break;
                    }
                }
                if(merge)
                    ufd_.merge(u, v);
            }
            this->fuseImpl(result);

            // evaluate if the result
            // is indeed better than each proposal
            // Iff the result is not better we
            // use the best proposal as a result
            auto eMin = std::numeric_limits<double>::infinity();
            auto eMinIndex = 0;
            for(auto i=0; i<proposals.size(); ++i){
                const auto p = proposals[i];
                const auto e = objective_.evalNodeLabels(*p);
                if(e < eMin){
                    eMin = e;
                    eMinIndex = i;
                }
            }
            const auto eResult = objective_.evalNodeLabels(*result);            
            if(eMin < eResult){
                for(auto node : graph_.nodes()){
                    result->operator[](node) = proposals[eMinIndex]->operator[](node);
                }
            }
        }

    private:
        template<class NODE_MAP>
        void fuseImpl(NODE_MAP * result){

            // dense relabeling
            std::set<uint64_t> relabelingSet;
            for(const auto node: graph_.nodes()){
                relabelingSet.insert(ufd_.find(node));
            }
            auto denseLabel = 0;
            for(auto sparse: relabelingSet){
                nodeToDense_[sparse] = denseLabel;
                ++denseLabel;
            }
            const auto numberOfNodes = relabelingSet.size();
            
            // build the graph
            FmGraph fmGraph(numberOfNodes);
            
            for(auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;
                const auto lu = nodeToDense_[ufd_.find(u)];
                const auto lv = nodeToDense_[ufd_.find(v)];
                NIFTY_CHECK_OP(lu,<,numberOfNodes,"");
                NIFTY_CHECK_OP(lv,<,numberOfNodes,"");
                if(lu != lv){
                    fmGraph.insertEdge(lu, lv);
                }
            }
            
            // setup objective
            FmObjective fmObjective(fmGraph);
            auto & fmWeights = fmObjective.weights();
            for(auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;
                const auto lu = nodeToDense_[ufd_.find(u)];
                const auto lv = nodeToDense_[ufd_.find(v)];
                if(lu != lv){
                    auto e = fmGraph.findEdge(lu, lv);
                    NIFTY_CHECK_OP(e,!=,-1,"");
                    fmWeights[e] += objective_.weights()[edge];
                }
            }

            // solve that thing
            
            
            auto solverPtr = settings_.mcFactory->createRawPtr(fmObjective);
            FmNodeLabels fmLabels(fmGraph);
            FmEmptyVisitor fmVisitor;
            solverPtr->optimize(fmLabels, &fmVisitor);
            delete solverPtr;

            
            for(auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;
                const auto lu = nodeToDense_[ufd_.find(u)];
                const auto lv = nodeToDense_[ufd_.find(v)];
                if(lu != lv){
                    if(fmLabels[lu] == fmLabels[lv]){
                        ufd_.merge(u, v);
                    }
                }
            }
            for(const auto node : graph_.nodes()){
                result->operator[](node)  = ufd_.find(node);
            }
        }


        const Objective & objective_;
        const Graph & graph_;
        Settings settings_;
        nifty::ufd::Ufd< > ufd_;
        NodeLabels nodeToDense_;
    };







} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_MULTICUT_FUSION_MOVE_HXX
