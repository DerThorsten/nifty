#pragma once

#include <iostream>
#include <string>
#include <random>
#include <mutex>

#include "nifty/parallel/threadpool.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_visitor_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_factory.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace multicut{

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



        enum NoiseType{
            UNIFORM_NOISE,
            NORMAL_NOISE,
            MAKE_LESS_CERTAIN
        };


        struct Settings{
            FactorySmartPtr mcFactory;
            size_t numberOfIterations{100};
            int numberOfThreads{-1};
            int verbose = 2;
            int seed = 42;
            NoiseType noiseType{UNIFORM_NOISE};
            double noiseMagnitude{1.0};
        };

        PerturbAndMap(const Objective & objective, const Settings settings = Settings());
        ~PerturbAndMap();


        const Objective & objective()const;
        const Graph & graph()const;

        void optimize(EdgeState & edgeState);

        void optimize(const NodeLabels & startingPoint, EdgeState & edgeState);


    private:

        struct ThreadData{
            ThreadData(const size_t threadId,const int seed, const Graph & graph)
            :   objective_(graph),
                solver_(nullptr),
                gen_(threadId+seed),
                distUniform01_(0,1),
                distNormal_(0,1){

            }

            InternalObjective  objective_;
            MulticutBaseType * solver_;
            std::mt19937 gen_;
            std::uniform_real_distribution<> distUniform01_;
            std::normal_distribution<> distNormal_;
        };


        template<class WEIGHTS>
        void perturbWeights(const size_t threadId, WEIGHTS & perturbedWeights);




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
                threadDataPtr = new ThreadData(tid, settings_.seed, graph_);
                // copy weights
                auto & weightsCopy = threadDataVec_[tid]->objective_.weights();
                for(const auto edge : this->graph().edges())
                    weightsCopy[edge] = weightsOriginal[edge];

                // allocate solver
                threadDataPtr->solver_ = settings_.mcFactory->create(threadDataPtr->objective_);
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

        //std::cout<<"optimize \n";
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

                //std::cout<<"perturb \n";
                this->perturbWeights(threadId, obj.weights());
                
                //std::cout<<"propergate weight change \n";
                solver->weightsChanged();

                //std::cout<<"set sp \n";
                NodeLabels arg(graph_);
                for(const auto node : graph_.nodes())
                    arg[node] = startingPoint[node];

                //std::cout<<"optimize \n";
                MulticutVerboseVisitor<Objective> v;
                solver->optimize(arg, nullptr);

                //std::cout<<"write res \n";
                mtx.lock();
                for(const auto edge : graph_.edges()){
                    const auto uv = graph_.uv(edge);
                    if(arg[uv.first] != arg[uv.second])
                        ++edgeCutCounter[edge];
                }
                ++nFinished;
                if(settings_.verbose >= 1){
                    std::cout<<nFinished<<" / "<<settings_.numberOfIterations<<"\n";
                }
                mtx.unlock();
                //std::cout<<"done\n";
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
        const size_t threadId,
        WEIGHTS & perturbedWeights
    ){

        auto & td = *threadDataVec_[threadId];
        auto & gen = td.gen_;
        const auto noiseMag = settings_.noiseMagnitude;
        const auto originalWeights = objective_.weights();


        if(settings_.noiseType == UNIFORM_NOISE){
            auto & d = td.distUniform01_;
            for(auto edge : graph_.edges()){
                perturbedWeights[edge] = originalWeights[edge] + (d(gen)-0.5)*2.0*noiseMag;
            }
        }
        else if(settings_.noiseType == NORMAL_NOISE){
            auto & d = td.distNormal_;
            for(auto edge : graph_.edges()){
                perturbedWeights[edge] = originalWeights[edge] + d(gen)*noiseMag;
            }
        }
        else if(settings_.noiseType == MAKE_LESS_CERTAIN){
            auto & d = td.distUniform01_;
            for(auto edge : graph_.edges()){

                const auto ow = originalWeights[edge];
                const auto sgn = ow < 0.0 ? -1.0 : 1.0;
                const auto rawNoise = d(gen);
                const auto noise = std::abs(ow)*-1.0*rawNoise*sgn*noiseMag;
                //std::cout<<"ow "<<ow<<" noise "<<noise<<"\n";
                perturbedWeights[edge] = originalWeights[edge] + noise;
            }
        }
    }

} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

