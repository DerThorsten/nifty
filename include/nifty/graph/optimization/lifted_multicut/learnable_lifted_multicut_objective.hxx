#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_HXX


#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/structured_learning/weight_vector.hxx"
#include "nifty/structured_learning/instances/weighted_edge.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{





    /**
     * @brief      Class for learnable lifted multicut objective.
     * 
     *
     * @tparam     GRAPH        The graph class
     * @tparam     WEIGHT_TYPE  float or double
     */
    template<class GRAPH, class WEIGHT_TYPE>   
    class LearnableLiftedMulticutObjective :  
        public LiftedMulticutObjective<GRAPH, WEIGHT_TYPE>
    {   
    public:
        typedef LiftedMulticutObjective<GRAPH, WEIGHT_TYPE> BaseType;
        typedef typename BaseType::GraphType       GraphType;
        typedef typename BaseType::LiftedGraphType LiftedGraphType;
        typedef typename BaseType::WeightType      WeightType;

        LearnableLiftedMulticutObjective(
            const GraphType & graph, 
            const size_t numberOfWeights,
            const int reserveAdditionalEdges = -1
        );


        template<class WEIGHT_INDICES_ITER, class FEATURE_ITER>
        std::pair<bool,uint64_t> addWeightedFeatures(const uint64_t u, const uint64_t v, 
                                 WEIGHT_INDICES_ITER, WEIGHT_INDICES_ITER, 
                                 FEATURE_ITER, const WeightType constTerm=0.0, 
                                 const bool overwriteConstTerm = false);

        template<class WEIGHT_VECTOR>
        inline void changeWeights(const WEIGHT_VECTOR & weightVector);
        
        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void getGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;

        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void addGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;

        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void substractGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;


    private:
        template<class NODE_LABELS, class GRADIENT_VECTOR, class BINARY_OPERATOR>
        inline void accumulateGradient(const NODE_LABELS & ,GRADIENT_VECTOR & ,BINARY_OPERATOR && )const;


        typedef structured_learning::instances::WeightedEdge<WEIGHT_TYPE> WeightedEdgeType;
        typedef typename GraphType:: template EdgeMap<WeightedEdgeType> WeightedEdgeCosts;

        WeightedEdgeCosts weightedEdgeCosts_;
        size_t numberOfWeights_;

    };


    template<class GRAPH, class WEIGHT_TYPE>
    LearnableLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    LearnableLiftedMulticutObjective(
        const GraphType & graph, 
        const size_t numberOfWeights,
        const int reserveAdditionalEdges
    )
    :   BaseType(graph,  reserveAdditionalEdges),
        weightedEdgeCosts_(this->liftedGraph()),
        numberOfWeights_(numberOfWeights){

    }


    template<class GRAPH, class WEIGHT_TYPE>
    template<class WEIGHT_INDICES_ITER, class FEATURE_ITER>
    std::pair<bool,uint64_t>  
    LearnableLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addWeightedFeatures(
        const uint64_t u, const uint64_t v,
        WEIGHT_INDICES_ITER weightIndicesBegin,  
        WEIGHT_INDICES_ITER weightIndicesEnd, 
        FEATURE_ITER featuresBegin, 
        const WeightType constTerm, 
        const bool overwriteConstTerm
    ){
        uint64_t edge;
        bool addedNewEdge;
        std::tie(edge, addedNewEdge) = this->setCost(u, v);

        if(addedNewEdge){
            // new WeightedEdge
            WeightedEdgeType weightedEdge;
            
            while(weightIndicesBegin != weightIndicesEnd){
                weightedEdge.addWeightedFeature(*weightIndicesBegin, *featuresBegin);
                ++weightIndicesBegin;
                ++featuresBegin;
            }
            weightedEdge.setConstTerm(constTerm,overwriteConstTerm);
            weightedEdgeCosts_.insertedEdge(edge, weightedEdge);
        }
        else{
            // existing edge
            auto & weightedEdge = weightedEdgeCosts_[edge];
            while(weightIndicesBegin != weightIndicesEnd){
                weightedEdge.addWeightedFeature(*weightIndicesBegin, *featuresBegin);
                ++weightIndicesBegin;
                ++featuresBegin;
            }
            weightedEdge.setConstTerm(constTerm,overwriteConstTerm);

        }
        return addedNewEdge;
    }

    template<class GRAPH, class WEIGHT_TYPE>
    template<class WEIGHT_VECTOR>
    void 
    LearnableLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    changeWeights(
        const WEIGHT_VECTOR & weightVector
    ){
        this->liftedGraph().forEachEdge([&](const uint64_t edge){
            this->weights_[edge] = weightedEdgeCosts_[edge].value(weightVector);
        });
    }



        
    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    LearnableLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    getGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        for(auto i=0; i<numberOfWeights_; ++i){
            gradient[0] = 0.0;
        }
        this->accumulateGradient(nodeLabels, gradient, 
            [](const float a, const float b){
                return a+b;
            }
        );
    }

    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    LearnableLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        this->accumulateGradient(nodeLabels, gradient, 
            [](const float a, const float b){
                return a+b;
            }
        );
    }

    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    LearnableLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    substractGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {

    }


    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR, class BINARY_OPERATOR>
    void 
    LearnableLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    accumulateGradient(
        const NODE_LABELS & nodeLabels,
        GRADIENT_VECTOR & gradient,
        BINARY_OPERATOR && binaryOperator
    )const{
        this->liftedGraph().forEachEdge([&](const uint64_t edge){
            const auto uv = this->uv(edge);
            if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                weightedEdgeCosts_[edge].accumulateGradient(gradient, binaryOperator);
            }
        });
    }


} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_HXX
