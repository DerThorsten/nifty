#pragma once
#ifndef NIFTY_STRUCTURED_LEARNING_INSTANCES_WEIGHTED_EDGE_HXX
#define NIFTY_STRUCTURED_LEARNING_INSTANCES_WEIGHTED_EDGE_HXX


#include "nifty/container/boost_flat_map.hxx"

namespace nifty{
namespace structured_learning{
namespace instances{




    /**
     * @brief      Class for linear weighted edges.
     *
     * @tparam     T  interal value type for storing values
     *                of weights and the features.
     *                Must be a floating point type.
     *                'float' or 'double' are the usual
     *                values.  
     *                
     *  
     */ 
    template<class T>
    class WeightedEdge{
    public:
        typedef T value_type;
        typedef T FeatureType;
        typedef T WeightType;
        typedef uint16_t WeightIndexType;


        void addWeightedFeature(const WeightIndexType index, const WeightType feature){
            indexFeatureMap_[index] = feature;
        }

        /**
         * @brief      accumulate gradient for cut edges
         *
         * @param      gradient        The gradient vector
         *
         * @tparam     WEIGHT_VECTOR        current weight vector
         * @tparam     GRADIENT_VECTOR      gradient vector to store the result
         * @tparam     BINARY_OPERATOR              binary operator like std::plus (gradient[i] = op(gradient[i], accumulatedGradient[i]))
         * 
         * Accumulate the gradient vector for a cut edge.
         * Important, if this edge is not cut, the gradient is zero!
         * 
         */
        template<class GRADIENT_VECTOR, class BINARY_OPERATOR>
        void accumulateGradient(
            GRADIENT_VECTOR & gradient,
            BINARY_OPERATOR && binaryOperator
        )const{
            for(const auto & indexFeaturePair : indexFeatureMap_){
                const auto wIndex = indexFeaturePair.first;
                const auto f = indexFeaturePair.second;
                gradient[wIndex] = binaryOperator(gradient[wIndex], f);
            }
        }

        /**
         * @brief      compute the current value of the edge given the weight vector
         *
         * @param[in]  weights        The weights vector
         *
         * @tparam     WEIGHT_VECTOR  class like nifty::structured_learning::WeightVector   
         *
         * @return     the current cut value of the edge
         */
        template<class WEIGHT_VECTOR>
        value_type value(
            const WEIGHT_VECTOR & weights
        )const{
            auto accVal = 0.0;
            for(const auto & indexFeaturePair : indexFeatureMap_){
                const auto wIndex = indexFeaturePair.first;
                const auto f = indexFeaturePair.second;
                accVal += f * weights[wIndex];
            }
            return accVal + constTerm_;
        }


        void setConstTerm(const WeightType constTerm){
            constTerm_ = constTerm;
        }

        void addConstTerm(const WeightType constTerm){
            constTerm_ += constTerm;
        }

    private:
        typedef std::pair<WeightIndexType, FeatureType> IndexFeature;
        //std::vector<IndexFeature> indexFeaturePairs_;
        container::BoostFlatMap<WeightIndexType,FeatureType> indexFeatureMap_;
        FeatureType constTerm_;
    };

}
}
}

#endif // NIFTY_STRUCTURED_LEARNING_INSTANCES_WEIGHTED_EDGE_HXX