#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace higher_order_multicut{

    
    template<class WEIGHT_TYPE = float>
    class HigherOrderFactor{
    public: 
        template<class SIZE_TYPE>
        HigherOrderFactor(std::initializer_list<SIZE_TYPE> shape)
        :   valueTable_(shape),
            


        size_t arity()const{
            return edgeIds_.size();
        }
        const std::vector<uint64_t> & edgeIds(){
            return edgeIds_;
        }

        const nifty::marray::Marray<WEIGHT_TYPE>  & valueTable(){
            return valueTable_;
        }
    private:

        nifty::marray::Marray<WEIGHT_TYPE> valueTable_;
        std::vector<uint64_t> edgeIds_;
    };

    template<class GRAPH, class WEIGHT_TYPE >   
    class HigherOrderMulticutObjective 
    {   
    public:
        typedef GRAPH GraphType;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabelsType;
        
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<GraphType, WeightType> WeightsMap;
        HigherOrderMulticutObjective(const GraphType & g )
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
        const std::vector<HigherOrderFactor> & higherOrderFactors() const{
            return higherOrderFactors_;
        }

    private:

        const GraphType & graph_;
        WeightsMap weights_;


        std::vector<HigherOrderFactor> higherOrderFactors_;
    };

} // namespace nifty::graph::opt::higher_order_multicut    
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

