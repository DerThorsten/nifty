#pragma once
#ifndef NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LOSS_AUGMENTED_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX
#define NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LOSS_AUGMENTED_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX

#include "nifty/graph/optimization/lifted_multicut/loss_augmented_view_lifted_multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{


    template<class OBJECTIVE>
    struct LiftedMulticutObjectiveName;





    template<class WEIGHTED_MODEL>
    struct LiftedMulticutObjectiveName<LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL> >{
        static std::string name(){

            const auto innerModelName = LiftedMulticutObjectiveName<WEIGHTED_MODEL>::name();

            return (std::string("LossAugmentedViewLiftedMulticutObjective") + innerModelName);
        }
    };

}
}
}

#endif /* NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LOSS_AUGMENTED_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX */
