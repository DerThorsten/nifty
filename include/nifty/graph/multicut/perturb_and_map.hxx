#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX
#define NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX

#include <iostream>
#include <string>
#include <random>

#include "nifty/graph/multicut/multicut_base.hxx"
#include "nifty/graph/multicut/multicut_visitor_base.hxx"
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
        typedef MulticutBase<Objective> MulticutBaseType;
        typedef MulticutFactoryBase<InternalObjective> InternalMcFactoryBase;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;
        typedef typename Graph:: template EdgeMap<double>   EdgeState;

        struct Settings{
            std::shared_ptr<InternalMcFactoryBase> mcFactory;
            size_t numberOfIterations{1000};
            size_t numberOfThreads{16};

        };

        PerturbAndMap(const Objective & objective, const Settings settings = Settings())
        :   objective_(objective),
            graph_(objective.graph()),
            settings_(settings),
            objectives_(),
            solvers_(){


            NIFTY_CHECK(bool(settings_.mcFactory),"factory must not be empty");

            // nthreads
            auto nThreads = settings_.numberOfThreads;
            const auto & weightsOriginal = objective_.weights();

            // create  objectives
            std::cout<<"create obj\n";
            objectives_.resize(nThreads);
            for(size_t i=0; i<objectives_.size(); ++i){
                objectives_[i] = new InternalObjective(this->graph());
                auto & weightsCopy = objectives_[i]->weights();
                for(const auto edge : this->graph().edges())
                    weightsCopy[edge] = weightsOriginal[edge];
            }

            // allocate  solvers
            std::cout<<"create solvers\n";
            solvers_.resize(nThreads);
            for(size_t i=0; i<solvers_.size(); ++i){
                const auto & obj = *objectives_[i];
                solvers_[i] = settings_.mcFactory->createRawPtr(obj);
            }
            std::cout<<"done\n";
        }

        ~PerturbAndMap(){
            for(size_t i=0; i<solvers_.size(); ++i){
                delete objectives_[i];
                delete solvers_[i];
            }
        }


        const Objective & objective()const{
            return objective_;
        }
        const Graph & graph()const{
            return objective_.graph();
        }

        void optimize(EdgeState & edgeState){
            NodeLabels startingPoint(this->graph());
            this->optimize(startingPoint, edgeState);
        }

        void optimize(const NodeLabels & startingPoint, EdgeState & edgeState){
            std::cout<<"in the opt\n";
            for(size_t i=0; i<settings_.numberOfIterations; ++i){
                std::cout<<"i "<<i<<"\n";
                auto obj = objectives_[0]; // fixme, atm fixed to first thread
                auto solver = solvers_[0]; // fixme, atm fixed to first thread
                this->perturbWeights(obj->weights());
                solver->weightsChanged();
                std::cout<<"fill nl\n";
                NodeLabels arg(graph_);
                for(const auto node : graph_.nodes()){
                    arg[node] = startingPoint[node];
                }
                MulticutVerboseVisitor<Objective> v;
                std::cout<<"optimize nl\n";
                solver->optimize(arg, &v);//nullptr);
                std::cout<<" arg pobj "<<obj->evalNodeLabels(arg)<<" oobj "<<objective_.evalNodeLabels(arg)<<"\n";
            }

        }

        template<class WEIGHTS>
        void perturbWeights( WEIGHTS & perturbedWeights){



            // fixme, all threads want to store their own randgen
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 10);

            const auto originalWeights = objective_.weights();
            for(auto edge : graph_.edges()){
                perturbedWeights[edge] = originalWeights[edge] + dis(gen);
            }
        }

 


    private:
        const Objective & objective_;
        const Graph & graph_;
        Settings settings_;

        std::vector<InternalObjective *> objectives_;
        std::vector<MulticutBaseType * > solvers_;
    };



} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX
