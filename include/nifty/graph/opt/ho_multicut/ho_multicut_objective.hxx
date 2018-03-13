#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"

#include "xtensor/xarray.hpp"

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{

    
    template<class WEIGHT_TYPE = float>
    class HigherOrderFactor{
    public: 
        typedef WEIGHT_TYPE value_type;


        template<class ARRAY, class VI>
        HigherOrderFactor(
            const ARRAY & array, 
            const VI & vi
        )
        :   valueTable_(array),
            edgeIds_(vi){

        }



        HigherOrderFactor()
        :   valueTable_(),
            edgeIds_(){

        }

        std::size_t arity()const{
            return edgeIds_.size();
        }

        const auto& edgeIds()const{
            return edgeIds_;
        }

        const auto  & valueTable()const{
            return valueTable_;
        }
    private:

        xt::xarray<value_type> valueTable_;
        std::vector<uint64_t> edgeIds_;
    };

    template<class GRAPH, class WEIGHT_TYPE >   
    class HoMulticutObjective 
    {   
    public:
        typedef GRAPH GraphType;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabelsType;
        
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<GraphType, WeightType> WeightsMap;
        HoMulticutObjective(const GraphType & g )
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
        const auto & higherOrderFactors() const{
            return higherOrderFactors_;
        }
        template<class ARRAY, class VI>
        void addHigherOrderFactor(
            const ARRAY & array, 
            const VI & vi
        )
        {
            higherOrderFactors_.emplace_back(array, vi);
        }

        template<class NODE_LABELS>
        double evalNodeLabels(const NODE_LABELS & nodeLabels)const{

            WeightType sum(0);

            // unaries
            for(const auto edge: graph_.edges()){
                const auto uv = graph_.uv(edge);
                
                if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                    sum += weights_[edge];
                }
            }

            for(const auto& f : higherOrderFactors_){
                const auto& vt = f.valueTable();
                const auto& edges = f.edgeIds();
            }
            return sum;
        }

    private:

        const GraphType & graph_;
        WeightsMap weights_;


        std::vector<HigherOrderFactor<WeightType>> higherOrderFactors_;
    };

} // namespace nifty::graph::opt::ho_multicut    
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

