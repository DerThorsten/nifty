#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX
#define NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX


#include "nifty/graph/multicut/multicut_factory.hxx"
#include "nifty/graph/multicut/multicut_objective.hxx"

namespace nifty {
namespace graph {


    template<class OBJECTIVE>
    class PerturbAndMap{
    public:
        typedef OBJECTIVE Objective;
        typedef typename  Objective::Graph Graph;
        typedef MulticutObjective<Graph, double> InternalObjective;
        typedef MulticutFactoryBase<InternalObjective> InternalMcFactoryBase;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;
        typedef typename Graph:: template EdgeMap<double>   EdgeState;

        struct Settings{
            std::shared_ptr<InternalMcFactoryBase> mcFactory;
            size_t numberOfIterations{1000};

        };

        PerturbAndMap(const Objective & objective, const Settings settings = Settings())
        :   objective_(objective),
            settings_(settings){
        }

        const Objective & objective()const{
            return objective_;
        }
        const Graph & graph()const{
            return objective_.graph();
        }

        void optimize(const NodeLabels & startingPoint, EdgeState & edgeState){
            std::cout<<"in the opt\n";
            
        }
        void optimize(EdgeState & edgeState){
            NodeLabels startingPoint(this->graph());
            this->optimize(startingPoint, edgeState);
        }


    private:
        const Objective & objective_;
        Settings settings_;
    };



} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX
