#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace mincut{

    template<class CHILD_OBJECTIVE, class GRAPH, class WEIGHT_TYPE>
    class MincutObjectiveBase{
    public:

        typedef CHILD_OBJECTIVE ChildObjective;
        typedef MincutObjectiveBase<ChildObjective, GRAPH, WEIGHT_TYPE> Self;

        template<class NODE_LABELS>
        WEIGHT_TYPE evalNodeLabels(const NODE_LABELS & nodeLabels)const{
            WEIGHT_TYPE sum = static_cast<WEIGHT_TYPE>(0.0);
            const auto & w = _child().weights();
            const auto & g = _child().graph();
            for(const auto edge: g.edges()){
                const auto uv = g.uv(edge);

                if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                    sum += w[edge];
                }
            }
            return sum;
        }
    private:
        ChildObjective & _child(){
           return *static_cast<ChildObjective *>(this);
        }
        const ChildObjective & _child()const{
           return *static_cast<const ChildObjective *>(this);
        }

    };


    template<class GRAPH, class WEIGHT_TYPE>
    class MincutObjective :  public
        MincutObjectiveBase<
            MincutObjective<GRAPH, WEIGHT_TYPE>, GRAPH, WEIGHT_TYPE
        >
    {
    public:
        typedef GRAPH GraphType;
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<GraphType, WeightType> WeightsMap;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabelsType;
        MincutObjective(const GraphType & g )
        :   graph_(g),
            weights_(g, 0.0)
        {

        }
        WeightsMap & weights(){
            return weights_;
        }

        // MUST IMPL INTERFACE
        const GraphType & graph() const{
            return graph_;
        }
        const WeightsMap & weights() const{
            return weights_;
        }
    private:
        const GraphType & graph_;
        WeightsMap weights_;
    };
} // namespace nifty::graph::opt::mincut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty
