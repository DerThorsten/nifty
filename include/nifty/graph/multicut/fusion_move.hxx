#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_FUSION_MOVE_HXX
#define NIFTY_GRAPH_MULTICUT_FUSION_MOVE_HXX

#include <mutex>          // std::mutex

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/multicut/multicut_base.hxx"
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

        struct Settings{

        };

        FusionMove(const Objective & objective, const Settings & settings = Settings())
        :   objective_(objective),
            graph_(objective.graph()),
            settings_(settings),
            ufd_(objective.graph().maxNodeId()+1),
            nodeToDense_(objective.graph())
        {

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
        }

    private:
        template<class NODE_MAP>
        void fuseImpl(NODE_MAP * result){

            // dense relabeling
            std::set<uint64_t> relabeling;
            for(const auto node: graph_.nodes()){
                relabeling.insert(ufd_.find(node));
            }
            auto denseLabel = 0;
            for(auto sparse: relabeling){
                nodeToDense_[sparse] = denseLabel;
                ++denseLabel;
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
