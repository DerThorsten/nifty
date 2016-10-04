#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_WEIGHTED_LIFTED_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_WEIGHTED_LIFTED_MULTICUT_OBJECTIVE_HXX


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
    class WeightedLiftedMulticutObjective :  
        public LiftedMulticutObjective<GRAPH, WEIGHT_TYPE>
    {   
    public:
        typedef LiftedMulticutObjective<GRAPH, WEIGHT_TYPE> BaseType;
        typedef typename BaseType::GraphType       GraphType;
        typedef typename BaseType::LiftedGraphType LiftedGraphType;
        typedef typename BaseType::WeightType      WeightType;

        WeightedLiftedMulticutObjective(
            const GraphType & graph,
            const int reserveAdditionalEdges = -1
        );


        template<class WEIGHT_INDICES_ITER, class FEATURE_ITER>
        std::pair<bool,uint64_t> addWeightedFeatures(const uint64_t u, const uint64_t v, 
                                 WEIGHT_INDICES_ITER, WEIGHT_INDICES_ITER, 
                                 FEATURE_ITER, const WeightType constTerm=0.0, 
                                 const bool overwriteConstTerm = false);



        std::pair<bool,uint64_t> addWeightedFeature(const uint64_t u, const uint64_t v, 
                                                    const uint64_t weightIndex, const WeightType feature);


        std::pair<bool,uint64_t> setConstTerm(const uint64_t u, const uint64_t v, const WeightType constTerm);
        std::pair<bool,uint64_t> addConstTerm(const uint64_t u, const uint64_t v, const WeightType constTerm);

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


        template<class F>
        std::pair<bool,uint64_t>  ensureEdge(const uint64_t u, const uint64_t v, F && f);

        typedef structured_learning::instances::WeightedEdge<WEIGHT_TYPE> WeightedEdgeType;
        typedef typename GraphType:: template EdgeMap<WeightedEdgeType> WeightedEdgeCosts;

        WeightedEdgeCosts weightedEdgeCosts_;

    };


    template<class GRAPH, class WEIGHT_TYPE>
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    WeightedLiftedMulticutObjective(
        const GraphType & graph, 
        const int reserveAdditionalEdges
    )
    :   BaseType(graph,  reserveAdditionalEdges),
        weightedEdgeCosts_(this->liftedGraph()){

    }


    template<class GRAPH, class WEIGHT_TYPE>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addWeightedFeature(
        const uint64_t u, 
        const uint64_t v, 
        const uint64_t weightIndex, 
        const WeightType feature
    ){
        return this->ensureEdge(u, v,
        [&](WeightedEdgeType & weightedEdge){
            weightedEdge.addWeightedFeature(weightIndex, feature);
        });
    }


    template<class GRAPH, class WEIGHT_TYPE>
    template<class WEIGHT_INDICES_ITER, class FEATURE_ITER>
    std::pair<bool,uint64_t>  
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addWeightedFeatures(
        const uint64_t u, const uint64_t v,
        WEIGHT_INDICES_ITER weightIndicesBegin,  
        WEIGHT_INDICES_ITER weightIndicesEnd, 
        FEATURE_ITER featuresBegin,
        const WeightType constTerm, 
        const bool overwriteConstTerm
    ){
        
        return this->ensureEdge(u, v,[&](WeightedEdgeType & weightedEdge){
            while(weightIndicesBegin != weightIndicesEnd){
                weightedEdge.addWeightedFeature(*weightIndicesBegin, *featuresBegin);
                ++weightIndicesBegin;
                ++featuresBegin;
            }
            if(overwriteConstTerm)
                weightedEdge.setConstTerm(constTerm);
            else
                weightedEdge.addConstTerm(constTerm);
        });
    }

    template<class GRAPH, class WEIGHT_TYPE>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    setConstTerm(
        const uint64_t u, 
        const uint64_t v, 
        const WeightType constTerm
    ){
        return this->ensureEdge(u, v,[&](WeightedEdgeType & weightedEdge){
            weightedEdge.setConstTerm(constTerm);
        }); 
    }

    template<class GRAPH, class WEIGHT_TYPE>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addConstTerm(
        const uint64_t u, 
        const uint64_t v, 
        const WeightType constTerm
    ){
        return this->ensureEdge(u, v,[&](WeightedEdgeType & weightedEdge){
            weightedEdge.addConstTerm(constTerm);
        }); 
    }



    template<class GRAPH, class WEIGHT_TYPE>
    template<class F>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    ensureEdge(
        const uint64_t u, 
        const uint64_t v, 
        F && f
    ){
        
        const auto ret = this->setCost(u, v);
        const uint64_t edge = ret.first;
        const bool addedNewEdge = ret.second;
        if(addedNewEdge){
            // new WeightedEdge
            WeightedEdgeType weightedEdge;
            weightedEdgeCosts_.insertedEdge(edge, weightedEdge);
            f(weightedEdgeCosts_[edge]);
        }
        else{
            // existing edge
            auto & weightedEdge = weightedEdgeCosts_[edge];
            f(weightedEdge);

        }
        return ret;
    }




    template<class GRAPH, class WEIGHT_TYPE>
    template<class WEIGHT_VECTOR>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
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
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    getGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        for(auto i=0; i<gradient.size(); ++i){
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
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
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
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    substractGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {

    }


    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR, class BINARY_OPERATOR>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
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

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_WEIGHTED_LIFTED_MULTICUT_OBJECTIVE_HXX
