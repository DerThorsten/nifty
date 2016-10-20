#pragma once

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_factory.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/loss_augmented_view_lifted_multicut_objective.hxx"



namespace nifty {
namespace structured_learning {


template<class WEIGHTED_OBJ>
class StructMaxMarginOracleLmc{
public:
    typedef WEIGHTED_OBJ WeightedObjectiveType;
    typedef nifty::graph::lifted_multicut::LossAugmentedViewLiftedMulticutObjective<WeightedObjectiveType> LossAugemntedObjectiveType;

    typedef nifty::graph::lifted_multicut::LiftedMulticutFactoryBase<LossAugemntedObjectiveType> SolverFactory;
    typedef std::shared_ptr<SolverFactory> SharedSolverFactory;
    typedef typename WEIGHTED_OBJ::GraphType GraphType;
    typedef typename GraphType:: template NodeMap<uint64_t> NodeLabels;
        

    StructMaxMarginOracleLmc(
        SharedSolverFactory solverFactory,
        const size_t numberOfWeights,
        const bool useWarmStart = true,
        int numberOfThreads = -1 // -1 mens 
    )
    :   solverFactory_(solverFactory),
        numberOfWeights_(numberOfWeights),
        useWarmStart_(useWarmStart)
    {
        
    }

    ~StructMaxMarginOracleLmc(){
        for(auto ptr : weightedObj_)
            delete ptr;
        for(auto ptr : lossAugmentedObj_)
            delete ptr;
    }


    virtual void getGradientAndValue(const std::vector<double> & weights, double & value, std::vector<double> & gradients){



        // init gradient with zeros
        for(auto & g : gradients ){
            g = 0;
        }


        value = 0.0;

        NIFTY_CHECK_OP(gradients.size(), ==, weights.size(),"");
        NIFTY_CHECK_OP(gradients.size(), ==, numberOfWeights_,"");
        NIFTY_CHECK_OP(weightedObj_.size(), ==, lossAugmentedObj_.size(),"");
        NIFTY_CHECK_OP(nodeGt_.size(), ==, lossAugmentedObj_.size(),"");

        for(auto i=0; i<lossAugmentedObj_.size(); ++i){

            auto & obj = *weightedObj_[i];
            auto & objL = *lossAugmentedObj_[i];


            //obj.changeWeights(weights);  // loss augmented model is calling change weights of obj
            objL.changeWeights(weights);

            const auto & nodeGt = nodeGt_[i];


            const auto c = obj.evalNodeLabels(nodeGt);



            auto f = [&](NodeLabels & mostViolated){
                auto solverPtr = solverFactory_->createRawPtr(objL);
                solverPtr->optimize(mostViolated, nullptr);
                delete solverPtr;
                value += c - objL.evalNodeLabels(mostViolated);
                obj.addGradient(nodeGt, gradients);
                obj.substractGradient(mostViolated, gradients);

            };


            if(!useWarmStart_){
                NodeLabels mostViolated(objL.graph());
                f(mostViolated);
            }
            else{
                auto & mostViolated = warmStartLabels_[i];
                f(mostViolated);
            }
        }



    }


  

    virtual size_t numberOfWeights()const{
        return numberOfWeights_;
    }


    template<class NODE_GT, class NODE_SIZES>
    void addModel(
        const GraphType & graph,
        const NODE_GT & nodeGt,
        const NODE_SIZES & nodeSizes
    ){

        auto obj = new WeightedObjectiveType(graph, this->numberOfWeights_);
        weightedObj_.push_back(obj);


        auto laObj = new LossAugemntedObjectiveType(*obj, nodeGt, nodeSizes);
        lossAugmentedObj_.push_back(laObj);

        nodeGt_.emplace_back(graph);
        auto &  gt = nodeGt_.back();

        for(auto node : graph.nodes())
            gt[node] = nodeGt[node];


        if(useWarmStart_){
            warmStartLabels_.emplace_back(graph);
            auto &  ws = warmStartLabels_.back();

            for(auto node : graph.nodes())
                ws[node] = 0;

        }

        

    }




 

    WeightedObjectiveType & getWeightedModel(const size_t index){
        return *(weightedObj_[index]);
    }

    LossAugemntedObjectiveType & getLossAugmentedModel(const size_t index){
        return *(lossAugmentedObj_[index]);
    }

    const size_t numberOfWeightesModels()const{
        return weightedObj_.size();
    }

    const size_t numberOfLossAugmentedModels()const{
        return lossAugmentedObj_.size();
    }


private:

    SharedSolverFactory solverFactory_;
    size_t numberOfWeights_;
    bool useWarmStart_;

    std::vector<NodeLabels> nodeGt_;
    std::vector<WeightedObjectiveType *>       weightedObj_;
    std::vector<LossAugemntedObjectiveType*>   lossAugmentedObj_;


    std::vector<NodeLabels> warmStartLabels_;

};




}
}