#pragma once



namespace nifty {
namespace structured_learning {


template<class WEIGHTED_OBJ, class LOSS_AUGMENTED_OBJ>
class StructMaxMarginOracleLmc{
public:
    StructMaxMarginOracleLmc(const size_t numberOfWeights)
    :   numberOfWeights_(numberOfWeights)
    {
        
    }


    virtual void getGradientAndValue(const std::vector<double> & weights, double & value, std::vector<double> & gradients){
        std::cout<<"get gradient and value\n";
    }

    virtual size_t numberOfWeights()const{
        return numberOfWeights_;
    }


    template <typename ...Params>
    void addWeightedModel(Params&&... params)
    {
        weightedObj_.emplace_back(std::forward<Params>(params)...);
    }

    template <typename ...Params>
    void addLossAugmentedModel(Params&&... params)
    {
        lossAugmentedObj_.emplace_back(std::forward<Params>(params)...);
    }


    WEIGHTED_OBJ & getWeightedModel(const size_t index){
        return weightedObj_[index];
    }

    LOSS_AUGMENTED_OBJ & getLossAugmentedModel(const size_t index){
        return lossAugmentedObj_[index];
    }

    const size_t numberOfWeightesModels()const{
        return weightedObj_.size();
    }

    const size_t numberOfLossAugmentedModels()const{
        return weightedObj_.size();
    }


private:


    size_t numberOfWeights_;

    std::vector<WEIGHTED_OBJ>       weightedObj_;
    std::vector<LOSS_AUGMENTED_OBJ> lossAugmentedObj_;

};




}
}