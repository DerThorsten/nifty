#pragma once
#ifndef NIFTY_GRAPH_GALA_GALA_HXX
#define NIFTY_GRAPH_GALA_GALA_HXX

#include <iostream>

#include "vigra/multi_array.hxx"
#include "vigra/priority_queue.hxx"


#include "nifty/tools/timer.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/gala/detail/contract_edge_callbacks.hxx"
#include "nifty/graph/gala/gala_classifier_rf.hxx"
#include "nifty/graph/gala/gala_feature_base.hxx"
#include "nifty/graph/gala/gala_instance.hxx"


#include "nifty/graph/multicut/multicut_base.hxx"
#include "nifty/graph/multicut/multicut_factory.hxx"
#include "nifty/graph/simple_graph.hxx"


namespace nifty{
namespace graph{


    struct GalaSettings{

        typedef nifty::graph::UndirectedGraph<> McOrderGraph;
        typedef nifty::graph::MulticutObjective<McOrderGraph, double> McOrderObjective;
        typedef nifty::graph::MulticutFactoryBase<McOrderObjective> McOrderFactoryBaseType;
        typedef std::shared_ptr<McOrderFactoryBaseType> McFactory;

        double threshold0{0.25};
        double threshold1{0.75};
        double thresholdU{0.25};
        uint64_t numberOfEpochs{3};
        uint64_t numberOfTrees{100};
        McFactory mapFactory;
        McFactory perturbAndMapFactory;
    };
    


    template<class GRAPH, class T, class CLASSIFIER = RfClassifier<T>  >
    class Gala{
    
    public:
        friend class detail_gala::TrainingCallback<GRAPH,T, CLASSIFIER> ;
        friend class detail_gala::TestCallback<GRAPH,T, CLASSIFIER> ;

        typedef GalaSettings Settings;
        typedef GRAPH GraphType;
        typedef CLASSIFIER ClassifierType;
        typedef detail_gala::TrainingCallback<GRAPH,T, CLASSIFIER> TrainingCallbackType;
        typedef detail_gala::TestCallback<GRAPH,T, CLASSIFIER> TestCallbackType;
        typedef  std::tuple<uint64_t,uint64_t,uint64_t,uint64_t> HashType;
   
        typedef GalaFeatureBase<GraphType, T>       FeatureBaseType;
        typedef Instance<GraphType, T>              InstanceType;
        typedef TrainingInstance<GraphType, T>      TrainingInstanceType;





        typedef nifty::graph::UndirectedGraph<> McOrderGraph;
        typedef nifty::graph::MulticutObjective<McOrderGraph, double> McOrderObjective;
        typedef nifty::graph::MulticutFactoryBase<McOrderObjective> McOrderFactoryBaseType;



        Gala(const Settings & settings = Settings());
        ~Gala();
        void addTrainingInstance(TrainingInstanceType & trainingInstance);
        void train();
        
        template<class NODE_LABELS>
        void predict(InstanceType & instance,  NODE_LABELS & nodeLabels)const;

    private:


        void trainInitalRf();
        void trainEpoch();

        template<class F>
        void discoveredExample(const F & features, const double pRf, const double pGt, const double uGt, const HashType & h);


        Settings trainingSettings_;

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
    :   trainingSettings_(settings),
        trainingCallbacks_(),
        classifier_(settings.numberOfTrees),
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
    void Gala<GRAPH,T,CLASSIFIER>::
    train(){
        tools::VerboseTimer t;
        t.startAndPrint("trainInitalRf");
        this->trainInitalRf();  
        t.stopAndPrint();
        t.reset();
        for(size_t i=0; i<trainingSettings_.numberOfEpochs; ++i){  

            t.startAndPrint("trainEpoch");
            this->trainEpoch();    
            t.stopAndPrint();
            t.reset();

            t.startAndPrint("classifier_.train()");
            classifier_.train(); 
            t.stopAndPrint();
            t.reset();

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
        if((pGt < trainingSettings_.threshold0 || pGt > trainingSettings_.threshold1) && uGt < trainingSettings_.thresholdU){
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
        TrainingInstanceType & trainingInstance
    ){
        auto cb = new TrainingCallbackType(trainingInstance,*this,trainingCallbacks_.size());
        trainingCallbacks_.push_back(cb);
    }      

    template<class GRAPH, class T, class CLASSIFIER>
    void 
    Gala<GRAPH,T,CLASSIFIER>::
    trainInitalRf(){

        //std::cout<<"Start to train gala\n";
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
                if((fgt < trainingSettings_.threshold0 || fgt> trainingSettings_.threshold1) && ufgt < trainingSettings_.thresholdU){
                    const uint8_t label = fgt < trainingSettings_.threshold0 ? 0 : 1;
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
                std::cout<<"#Edges "<<nEdges<<" #PQ "<<pq.size()<<" top "<<pq.topPriority()<<" #Nodes "<<nNodes<<" #ufd "<< contractionGraph.ufd().numberOfSets()<<"\n";
                if(pq.topPriority() > 10.0){
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
                        if(fgt < trainingSettings_.threshold0 && ugt < trainingSettings_.thresholdU){
                            contractionGraph.contractEdge(edge);
                        }
                        else{
                            std::cout<<"policy is wrong\n";
                            pq.push(edge, std::numeric_limits<T>::infinity());
                            trainingCallback->currentProbability_[edge] = std::numeric_limits<T>::infinity();
                        }
                    }   
                }
            }
        }
    }




    template<class GRAPH, class T, class CLASSIFIER>
    template<class NODE_LABELS>
    void 
    Gala<GRAPH,T,CLASSIFIER>::
    predict(
        InstanceType & instance,
        NODE_LABELS & labels
    )const{

        TestCallbackType callback(instance, *this);

        const auto & graph = callback.graph();
        auto features  = callback.features();
        const auto & pq  = callback.pq_;
        auto & contractionGraph = callback.contractionGraph_;
        // do the initial prediction
        callback.initalPrediction();

        //std::cout<<"start to predict\n";
        while(pq.topPriority()<0.5 && !pq.empty()){

            std::cout<<contractionGraph.numberOfNodes()<<" "<<graph.numberOfNodes()<<"\n";
            std::vector<uint64_t> edgesToContract;
            bool isDone = callback.toContract(edgesToContract);
            if(isDone){
                std::cout<<"done\n";
                break;
            }
            for(const auto edge : edgesToContract){
                //if(pq.topPriority()<0.5){
                if(pq.contains(edge))
                    contractionGraph.contractEdge(edge);
                //}
                //else{
                //    break;
                //}   
            }
        }

        for(const auto node : graph.nodes()){
            labels[node] = contractionGraph.ufd().find(node);
        }
    }  


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_HXX
