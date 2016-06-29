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

        struct Settings{
            std::shared_ptr<InternalMcFactoryBase> mcFactory;
        };

        PerturbAndMap(const Objective & objective, const Settings settings = Settings())
        :   objective_(objective),
            settings_(settings){
        }
    private:
        const Objective & objective_;
        Settings settings_;
    };



} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX
