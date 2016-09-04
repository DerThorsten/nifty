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


#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_factory.hxx"
#include "nifty/graph/undirected_list_graph.hxx"


namespace nifty{
namespace graph{


    


    template<class GRAPH, class T, class CLASSIFIER = RfClassifier<T>  >
    class Gala{
    
    public:
        
        friend class detail_gala::CallbackBase<GRAPH,T, CLASSIFIER, TrainingInstance<GRAPH, T>,detail_gala::TrainingCallback<GRAPH,T, CLASSIFIER  > >;
        friend class detail_gala::CallbackBase<GRAPH,T, CLASSIFIER, Instance<GRAPH, T>,detail_gala::TestCallback<GRAPH,T, CLASSIFIER  > >;
        friend class detail_gala::TrainingCallback<GRAPH,T, CLASSIFIER> ;
        friend class detail_gala::TestCallback<GRAPH,T, CLASSIFIER> ;

        typedef GRAPH GraphType;
        typedef CLASSIFIER ClassifierType;
        typedef detail_gala::TrainingCallback<GRAPH,T, CLASSIFIER> TrainingCallbackType;
        typedef detail_gala::TestCallback<GRAPH,T, CLASSIFIER> TestCallbackType;
        typedef typename TrainingCallbackType::ContractionOrderSettings ContractionOrderSettings;
        


        struct Settings{

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

            ContractionOrderSettings contractionOrderSettings;
        };


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
        void predict(InstanceType & instance,  NODE_LABELS & nodeLabels);

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
        //std::cout<<"discoveredExample\n";
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
        auto cb = new TrainingCallbackType(trainingInstance, trainingSettings_.contractionOrderSettings, *this,trainingCallbacks_.size());
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
        auto & someTrainingInstance =  trainingCallbacks_.front()->getInstance();
        const uint64_t numberOfFeatures = someTrainingInstance.numberOfFeatures();

        classifier_.initialize(numberOfFeatures);
        std::vector<T> fBuffer(numberOfFeatures);

        for(auto trainingCallback : trainingCallbacks_){  
            const auto & graph = trainingCallback->graph();
            auto features  = trainingCallback->features();
            auto & trainingInstance = trainingCallback->getInstance();

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


            auto & contractionGraph = trainingCallback->contractionGraph_;
        

            while(!trainingCallback->stopContraction() ){

                const auto nEdges = contractionGraph.numberOfEdges();
                const auto nNodes = contractionGraph.numberOfNodes();
                std::cout<<"#Edges "<<nEdges<<" #Nodes "<<nNodes<<" \n";



                if(nEdges == 0 || nNodes <=1){
                    break;
                }

                const auto toContract = trainingCallback->edgeToContractNext();
                
                // fetch the gt 
                const auto fgt = trainingCallback->getInstance().edgeGt()[toContract];
                const auto ugt = trainingCallback->getInstance().edgeGtUncertainty()[toContract];
                if(fgt < trainingSettings_.threshold0 && ugt < trainingSettings_.thresholdU){
                    contractionGraph.contractEdge(toContract);
                }
                else{
                    std::cout<<"policy is wrong\n";
                    trainingCallback->contractionOrder_.constraintsEdge(toContract);
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
    ){

        TestCallbackType callback(instance,trainingSettings_.contractionOrderSettings, *this);

        const auto & graph = callback.graph();
        auto features  = callback.features();
        auto & contractionGraph = callback.contractionGraph_;
        // do the initial prediction
        callback.initalPrediction();

        //std::cout<<"start to predict\n";
        while(!callback.stopContraction()){

            std::cout<<contractionGraph.numberOfNodes()<<" "<<graph.numberOfNodes()<<"\n";
            const auto toContract = callback.edgeToContractNext();
            contractionGraph.contractEdge(toContract);

        }

        for(const auto node : graph.nodes()){
            labels[node] = contractionGraph.ufd().find(node);
        }
    }  


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_HXX
