#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX
#define NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX

#include <iostream>
#include <string>
#include <random>
#include <mutex>

#include "nifty/parallel/threadpool.hxx"
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
        typedef std::shared_ptr<InternalMcFactoryBase> FactorySmartPtr;
        struct Settings{
            FactorySmartPtr mcFactory;
            size_t numberOfIterations{1000};
            int numberOfThreads{-1};
            int verbose = 2;
        };

        PerturbAndMap(const Objective & objective, const Settings settings = Settings());
        ~PerturbAndMap();


        const Objective & objective()const;
        const Graph & graph()const;

        void optimize(EdgeState & edgeState);

        void optimize(const NodeLabels & startingPoint, EdgeState & edgeState);


    private:

        struct ThreadData{
            ThreadData(const size_t threadId, const Graph & graph)
            :   objective_(graph),
                solver_(nullptr),
                gen_(threadId),
                dis_(0,10){

            }

            InternalObjective  objective_;
            MulticutBaseType * solver_;
            std::mt19937 gen_;
            std::uniform_real_distribution<> dis_;

        };


        template<class WEIGHTS>
        void perturbWeights( WEIGHTS & perturbedWeights);




        const Objective & objective_;
        const Graph & graph_;
        Settings settings_;

        std::vector<ThreadData*> threadDataVec_;
    };


    template<class OBJECTIVE>
    PerturbAndMap<OBJECTIVE>::
    PerturbAndMap(
        const Objective & objective, 
        const Settings settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        settings_(settings),
        threadDataVec_(){

        nifty::parallel::ParallelOptions popt(settings_.numberOfThreads);
        threadDataVec_.resize(popt.getActualNumThreads(),nullptr);

        NIFTY_CHECK(bool(settings_.mcFactory),"factory must not be empty");
        
        // nthreads
        auto nThreads = threadDataVec_.size();
        const auto & weightsOriginal = objective_.weights();


        nifty::parallel::parallel_foreach(settings_.numberOfThreads,
            popt.getActualNumThreads(),
            [&](size_t threadId, int tid){
                auto & threadDataPtr = threadDataVec_[tid];
                threadDataPtr = new ThreadData(tid, graph_);
                // copy weights
                auto & weightsCopy = threadDataVec_[tid]->objective_.weights();
                for(const auto edge : this->graph().edges())
                    weightsCopy[edge] = weightsOriginal[edge];

                // allocate solver
                threadDataPtr->solver_ = settings_.mcFactory->createRawPtr(threadDataPtr->objective_);
            }
        );

    }

    template<class OBJECTIVE>
    PerturbAndMap<OBJECTIVE>::
    ~PerturbAndMap(){
        for(size_t i=0; i<threadDataVec_.size(); ++i){
            delete threadDataVec_[i]->solver_;
            delete threadDataVec_[i];
        }
    }

    template<class OBJECTIVE>
    const typename PerturbAndMap<OBJECTIVE>::Objective & 
    PerturbAndMap<OBJECTIVE>::
    objective()const{
        return objective_;
    }

    template<class OBJECTIVE>
    const typename PerturbAndMap<OBJECTIVE>::Graph & 
    PerturbAndMap<OBJECTIVE>::
    graph()const{
        return objective_.graph();
    }

    template<class OBJECTIVE>
    void PerturbAndMap<OBJECTIVE>::
    optimize(
        EdgeState & edgeState
    ){
        NodeLabels startingPoint(this->graph());
        this->optimize(startingPoint, edgeState);
    }

    template<class OBJECTIVE>
    void PerturbAndMap<OBJECTIVE>::
    optimize(
        const NodeLabels & startingPoint, 
        EdgeState & edgeState
    ){

        std::cout<<"optimize \n";
        std::mutex mtx;
        auto nFinished = 0;

        typedef typename Graph:: template EdgeMap<uint64_t> EdgeCutCounter;
        EdgeCutCounter edgeCutCounter(graph_, 0);

        nifty::parallel::parallel_foreach(settings_.numberOfThreads,
            settings_.numberOfIterations,
            [&](size_t threadId, int items){
                auto & threadData = *threadDataVec_[threadId];
                auto & obj = threadData.objective_;
                auto solver = threadData.solver_;

                this->perturbWeights(obj.weights());
                solver->weightsChanged();
                NodeLabels arg(graph_);
                for(const auto node : graph_.nodes())
                    arg[node] = startingPoint[node];
                solver->optimize(arg, nullptr);

                mtx.lock();
                for(const auto edge : graph_.edges()){
                    const auto uv = graph_.uv(edge);
                    if(arg[uv.first] != arg[uv.second])
                        ++edgeCutCounter[edge];
                }
                ++nFinished;
                std::cout<<nFinished<<" / "<<settings_.numberOfIterations<<"\n";
                mtx.unlock();
            }
        );
        for(const auto edge : graph_.edges()){
            edgeState[edge] = double(edgeCutCounter[edge])/double(settings_.numberOfIterations);
        }
    }

    template<class OBJECTIVE>
    template<class WEIGHTS>
    void PerturbAndMap<OBJECTIVE>::
    perturbWeights( 
        WEIGHTS & perturbedWeights
    ){



        // fixme, all threads want to store their own randgen
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);

        const auto originalWeights = objective_.weights();
        for(auto edge : graph_.edges()){
            perturbedWeights[edge] = originalWeights[edge] + dis(gen);
        }
    }

















} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_PERTURB_AND_MAP_HXX
