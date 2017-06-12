#pragma once

#include <mutex>          // std::mutex
#include <memory>
#include <unordered_set>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_greedy_additive.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/common/solver_factory.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{


    /**
     * @brief      Fuse multiple labels into single one w.r.t. a LiftedMulticut objective
     *
     * @tparam     OBJECTIVE  LiftedMulticutObjective
     * 
     * 
     * \todo This implementation should only be used for sparse graphs
     * (atm only sparse graphs are implemented so this is not an issue so far)
     */
    template<class OBJECTIVE>
    class FusionMove{
    public:
        typedef OBJECTIVE Objective;
        typedef typename Objective::GraphType GraphType;
        typedef typename Objective::LiftedGraphType LiftedGraphType;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabels;
        

        typedef UndirectedGraph<> FmGraphType;
        typedef UndirectedGraph<> FmLiftedGraphType;
        typedef LiftedMulticutObjective<FmGraphType, double> FmObjective;
        typedef LiftedMulticutBase<FmObjective> FmLmcBase;
        typedef nifty::graph::optimization::common::SolverFactoryBase<FmLmcBase> FmLmcFactoryBase;
        typedef typename  FmLmcBase::NodeLabelsType FmNodeLabelsType;

        struct SettingsType{
            std::shared_ptr<FmLmcFactoryBase> lmcFactory;
        };

        FusionMove(const Objective & objective, const SettingsType & settings = SettingsType())
        :   objective_(objective),
            graph_(objective.graph()),
            liftedGraph_(objective_.liftedGraph()),
            settings_(settings),
            ufd_(objective.graph().nodeIdUpperBound()+1),
            nodeToDense_(objective.graph())
        {
            if(!bool(settings_.lmcFactory)){
                typedef LiftedMulticutGreedyAdditive<FmObjective> FmSolver;
                typedef nifty::graph::optimization::common::SolverFactory<FmSolver> FmFactory;
                settings_.lmcFactory = std::make_shared<FmFactory>();
            }
        }

        template<class NODE_MAP>
        void fuse(
            std::initializer_list<const NODE_MAP *> proposals,
            NODE_MAP * result
        ){
            std::vector<const NODE_MAP *> p(proposals);
            fuse(p, result);
        }


        template<class NODE_MAP >
        void fuse(
            const std::vector< const NODE_MAP *> & proposals,
            NODE_MAP * result 
        ){
            ufd_.reset();

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
            std::unordered_set<uint64_t> relabelingSet;
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
            FmGraphType       fmGraph(numberOfNodes);
            

            for(auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;
                const auto lu = nodeToDense_[ufd_.find(u)];
                const auto lv = nodeToDense_[ufd_.find(v)];
                if(lu != lv){
                    fmGraph.insertEdge(lu, lv);
                }
            }

            FmObjective fmObjective(fmGraph);

            for(auto edge : liftedGraph_.edges()){
                const auto uv = liftedGraph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;
                const auto lu = nodeToDense_[ufd_.find(u)];
                const auto lv = nodeToDense_[ufd_.find(v)];
                if(lu != lv){
                    fmObjective.setCost(lu,lv,objective_.weights()[edge],false);
                }
            }



            const auto fmEdges = fmGraph.numberOfEdges();

            if(fmEdges == 0){
                for(const auto node : graph_.nodes()){
                    result->operator[](node)  = ufd_.find(node);
                }
            }
            else{

                NIFTY_CHECK_OP(fmGraph.numberOfEdges(),>,0,"");

                auto solverPtr = settings_.lmcFactory->create(fmObjective);
                FmNodeLabelsType fmLabels(fmGraph);
                solverPtr->optimize(fmLabels, nullptr);
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
        }


        const Objective & objective_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        SettingsType settings_;
        nifty::ufd::Ufd< > ufd_;
        NodeLabels nodeToDense_;
    };





} // namespace nifty::graph::lifted_multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

