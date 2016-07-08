#pragma once
#ifndef NIFTY_GRAPH_GALA_GALA_HXX
#define NIFTY_GRAPH_GALA_GALA_HXX

#include <iostream>

#include "vigra/multi_array.hxx"
#include "vigra/priority_queue.hxx"

#include "nifty/graph/edge_contraction_graph.hxx"


#include "nifty/graph/gala/detail/contract_edge_callbacks.hxx"
#include "nifty/graph/gala/gala_classifier_rf.hxx"
#include "nifty/graph/gala/gala_feature_base.hxx"
#include "nifty/graph/gala/gala_instance.hxx"

namespace nifty{
namespace graph{


    template<class GRAPH, class T, class CLASSIFIER = RfClassifier<T>  >
    class Gala{
    
    public:
        friend class detail_gala::TrainingCallback<GRAPH,T, CLASSIFIER> ;

        typedef GRAPH GraphType;
        typedef CLASSIFIER ClassifierType;
        typedef detail_gala::TrainingCallback<GRAPH,T, CLASSIFIER> TrainingCallbackType;
        typedef  std::tuple<uint64_t,uint64_t,uint64_t,uint64_t> HashType;
   
        typedef GalaFeatureBase<GraphType, T>       FeatureBaseType;
        typedef Instance<GraphType, T>              InstanceType;
        typedef TrainingInstance<GraphType, T>      TrainingInstanceType;

        struct Settings{
            double threshold0_{0.25};
            double threshold1_{0.75};
            double thresholdU_{0.25};
        };
        

        Gala(const Settings & settings = Settings());
        ~Gala();
        void addTrainingInstance(TrainingInstanceType * trainingInstance);
        void train();
        
    private:

        void trainInitalRf();
        void trainEpoch();

        template<class F>
        void discoveredExample(const F & features, const double pRf, const double pGt, const double uGt, const HashType & h);


        Settings settings_;

        // training edge contraction callbacks and graphs
        std::vector<TrainingCallbackType *> trainingCallbacks_;

        ClassifierType classifier_;
        std::set<HashType> addedExamples_;
    };

    template<class GRAPH, class T, class CLASSIFIER>
    Gala<GRAPH,T,CLASSIFIER>::
    Gala(
        const Settings & settings
    )
    :   settings_(settings),
        trainingCallbacks_(),
        classifier_(),
        addedExamples_(){
    }

    template<class GRAPH, class T, class CLASSIFIER>
    Gala<GRAPH,T,CLASSIFIER>::
    ~Gala(){
        for(size_t i=0; i<trainingCallbacks_.size(); ++i){
            delete trainingCallbacks_[i];
        }
    }



    template<class GRAPH, class T, class CLASSIFIER>
    void Gala<GRAPH,T,CLASSIFIER>::train(){
        this->trainInitalRf();  
        for(size_t i=0; i<20; ++i){  
            this->trainEpoch();    
            classifier_.train(); 
        }
    }

    template<class GRAPH, class T, class CLASSIFIER>
    template<class F>
    void  Gala<GRAPH,T,CLASSIFIER>::
    discoveredExample(
        const F & features, 
        const double pRf, 
        const double pGt, 
        const double uGt, 
        const HashType & h
    ){
        // check if the training example is credible
        if((pGt < settings_.threshold0_ || pGt > settings_.threshold1_) && uGt < settings_.thresholdU_){
            if(addedExamples_.find(h) == addedExamples_.end()){
                addedExamples_.insert(h);
                classifier_.addTrainingExample(features, pGt > 0.5 ? 1 : 0);
            }
        }
    }

    template<class GRAPH, class T, class CLASSIFIER>
    void 
    Gala<GRAPH,T,CLASSIFIER>::
    addTrainingInstance(
        TrainingInstanceType * trainingInstance
    ){
        auto cb = new TrainingCallbackType(*trainingInstance,*this,trainingCallbacks_.size());
        trainingCallbacks_.push_back(cb);
    }      

    template<class GRAPH, class T, class CLASSIFIER>
    void 
    Gala<GRAPH,T,CLASSIFIER>::
    trainInitalRf(){

        std::cout<<"Start to train gala\n";
        const uint64_t numberOfGraphs = trainingCallbacks_.size();
        NIFTY_CHECK_OP(numberOfGraphs,>,0, "training set must not be empty");


        // fetch some information from an arbitrary training example
        auto & someTrainingInstance =  trainingCallbacks_.front()->trainingInstance_;
        const uint64_t numberOfFeatures = someTrainingInstance.numberOfFeatures();

        classifier_.initialize(numberOfFeatures);
        std::vector<T> fBuffer(numberOfFeatures);

        for(auto trainingCallback : trainingCallbacks_){  
            const auto & graph = trainingCallback->graph();
            auto features  = trainingCallback->features();
            auto & trainingInstance = trainingCallback->trainingInstance_;

            const auto & edgeGt = trainingInstance.edgeGt();
            const auto & edgeGtUncertainty = trainingInstance.edgeGtUncertainty();
            for(const auto edge : graph.edges()){
                auto fgt = edgeGt[edge];
                auto ufgt = edgeGtUncertainty[edge];
                if((fgt < settings_.threshold0_ || fgt> settings_.threshold1_) && ufgt < settings_.thresholdU_){
                    const uint8_t label = fgt < settings_.threshold0_ ? 0 : 1;
                    features->getFeatures(edge, fBuffer.data());
                    classifier_.addTrainingExample(fBuffer.data(), label);                
                }
            }
        }
        classifier_.train();
    }

    template<class GRAPH, class T, class CLASSIFIER>
    void 
    Gala<GRAPH,T,CLASSIFIER>::
    trainEpoch(){
        // do the initial prediction with the current random forest
        for(auto trainingCallback : trainingCallbacks_){  
            trainingCallback->reset();
        }
        // do the initial prediction on initial graph and initial features
        for(auto trainingCallback : trainingCallbacks_){  
            trainingCallback->initalPrediction();
        }
        for(auto trainingCallback : trainingCallbacks_){  

            auto & pq = trainingCallback->pq_;
            auto & contractionGraph = trainingCallback->contractionGraph_;
        
            while(!pq.empty() ){
                const auto nEdges = contractionGraph.numberOfEdges();
                const auto nNodes = contractionGraph.numberOfNodes();
                //std::cout<<"#Edges "<<nEdges<<" #PQ "<<pq.size()<<" top "<<pq.topPriority()<<" #Nodes "<<nNodes<<" #ufd "<< contractionGraph.ufd().numberOfSets()<<"\n";
                if(pq.topPriority() > 100.0){
                    break; 
                }
                if(nEdges == 0 || nNodes <=1){
                    break;
                }
                std::vector<uint64_t> edgesToContract;
                trainingCallback->toContract(edgesToContract);
                for(const auto edge : edgesToContract){
                    if(pq.contains(edge)){

                        // fetch the gt 
                        const auto fgt = trainingCallback->trainingInstance_.edgeGt()[edge];
                        const auto ugt = trainingCallback->trainingInstance_.edgeGtUncertainty()[edge];
                        if(fgt < settings_.threshold0_ && ugt < settings_.thresholdU_){
                            contractionGraph.contractEdge(edge);
                        }
                        else{
                            //std::cout<<"policy is wrong\n";
                            pq.push(edge, 10000.0);
                        }
                    }   
                }
            }
        }
    }



} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_HXX
