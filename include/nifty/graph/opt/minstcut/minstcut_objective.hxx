#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{

    template<class CHILD_OBJECTIVE, class GRAPH, class WEIGHT_TYPE>
    class MinstcutObjectiveBase{
    public:

        typedef CHILD_OBJECTIVE ChildObjective;
        typedef MinstcutObjectiveBase<ChildObjective, GRAPH, WEIGHT_TYPE> Self;

        template<class NODE_LABELS>
        WEIGHT_TYPE evalNodeLabels(const NODE_LABELS & nodeLabels)const{
            WEIGHT_TYPE sum = static_cast<WEIGHT_TYPE>(0.0);
            const auto & w = _child().weights();
            const auto & g = _child().graph();
            const auto & u = _child().unaries();

            for(const auto edge: g.edges()){
                const auto uv = g.uv(edge);
                
                if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                    sum += w[edge];
                }
            }
            for(auto node : g.nodes()){
                const auto l = nodeLabels[node];
                const auto & costPair = u[node];
                if(l == 0){
                    sum += costPair.first;
                }
                else if(l ==1){
                    sum += costPair.second;
                }
                else{
                    throw std::runtime_error("error - ???");
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
    class MinstcutObjective :  public
        MinstcutObjectiveBase<
            MinstcutObjective<GRAPH, WEIGHT_TYPE>, GRAPH, WEIGHT_TYPE
        >
    {   
    public:
        typedef GRAPH GraphType;
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<GraphType, WeightType> WeightsMap;
        typedef graph_maps::NodeMap<GraphType, std::pair<float, float> > UnaryMap;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabelsType;
        MinstcutObjective(const GraphType & g )
        :   graph_(g),
            weights_(g, 0.0),unaries_(g, {0.0 , 0.0})   
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
        const UnaryMap & unaries() const{
            return unaries_;
        }

        UnaryMap & unaries() {
            return unaries_;
        }


    private:
        const GraphType & graph_;
        WeightsMap weights_;
        UnaryMap unaries_;
    };
} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty
