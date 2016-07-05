#pragma once
#ifndef NIFTY_GRAPH_GALA_GALA_HXX
#define NIFTY_GRAPH_GALA_GALA_HXX

#include <iostream>

#include "vigra/multi_array.hxx"
#include "vigra/random_forest.hxx"
#include "vigra/priority_queue.hxx"

#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/gala/gala_feature_base.hxx"
#include "nifty/graph/gala/gala_instance.hxx"

namespace nifty{
namespace graph{


    template<class GRAPH, class T>
    class Gala;


    // also the training callback
    template<class GRAPH, class T>
    struct TrainingInstanceData{
        
        typedef GRAPH GraphType;
        typedef TrainingInstanceData<GraphType, T> Self;
        typedef TrainingInstance<GraphType, T>     TrainingInstanceType;
        typedef GalaFeatureBase<GraphType, T>     FeatureBaseType;

        typedef Gala<GraphType, T> GalaType;
        typedef EdgeContractionGraph<GraphType, Self>   TrainingEdgeContractionGraphType;
        typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

        TrainingInstanceData(TrainingInstanceType & trainingInstance, GalaType & gala)
        :   trainingInstance_(trainingInstance),
            contractionGraph_(trainingInstance.graph(), *this),
            pq_(trainingInstance.graph().maxEdgeId()+1),
            gala_(gala){
        }

        const GraphType & graph()const{
            return trainingInstance_.graph();
        }

        FeatureBaseType * features(){
            return trainingInstance_.features();
        }
        const uint64_t numberOfFeatures()const{
            return trainingInstance_.numberOfFeatures();
        }

        void contractEdge(const uint64_t edgeToContract){
            NIFTY_TEST(pq_.contains(edgeToContract));
            pq_.deleteItem(edgeToContract);
        }

        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
           trainingInstance_.features()->mergeNodes(aliveNode, deadNode);
        }

        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
            NIFTY_TEST(pq_.contains(aliveEdge));
            NIFTY_TEST(pq_.contains(deadEdge));
            trainingInstance_.features()->mergeEdges(aliveEdge, deadEdge);
            pq_.deleteItem(deadEdge);
           
        }

        void contractEdgeDone(const uint64_t edgeToContract){
            // recompute features  
            const auto u = contractionGraph_.nodeOfDeadEdge(edgeToContract);
            for(auto adj :contractionGraph_.adjacency(u)){
                const auto edge = adj.edge();
                this->recomputeFeaturesAndPredict(edge);
            }
        }

        void recomputeFeaturesAndPredict(const uint64_t edgeToUpdate){ 

            const auto nf = this->numberOfFeatures();
            vigra::MultiArray<2, T>       f(vigra::Shape2(1,nf));     
            vigra::MultiArray<2, double> p(vigra::Shape2(1,2));
            this->features()->getFeatures(edgeToUpdate, &f(0,0));

            for(size_t fi=0; fi<nf; ++fi){
                //std::cout<<f(fi)<<" ";
            }
            //std::cout<<"\n";
            gala_.classifier_->predictProbabilities(f,p);
            
            std::cout<<"edge "<<p(0,1)<<"\n";

            pq_.push(edgeToUpdate, p(0,1));
        }

        void toContract(std::vector<uint64_t> & toContract){
            toContract.resize(0);
            toContract.push_back(pq_.top());
        }

        TrainingInstanceType & trainingInstance_;
        TrainingEdgeContractionGraphType contractionGraph_;
        QueueType pq_;
        GalaType & gala_;
    };


    template<class GRAPH, class T>
    class Gala{
    
    public:
        friend class TrainingInstanceData<GRAPH,T> ;
        typedef TrainingInstanceData<GRAPH,T > TrainingInstanceDataType;

        struct Settings{
            double threshold0_{0.25};
            double threshold1_{0.75};
            double thresholdU_{0.25};
        };
        Gala(const Settings & settings = Settings())
        :   settings_(settings),
            trainingSet_(),
            classifier_(nullptr){
        }

        ~Gala(){
            for(size_t i=0; i<trainingSetData_.size(); ++i){
                delete trainingSetData_[i];
            }
        }

        typedef GRAPH GraphType;
        typedef GalaFeatureBase<GraphType, T>       FeatureBaseType;
        typedef Instance<GraphType, T>              InstanceType;
        typedef TrainingInstance<GraphType, T>      TrainingInstanceType;
        typedef GRAPH Graph;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;


        void addTrainingInstance(TrainingInstanceType * trainingInstance){
            trainingSet_.push_back(trainingInstance);
            trainingSetData_.push_back(new TrainingInstanceDataType(*trainingInstance,*this));
        }        

        void train();
        void trainInitalRf();
        void trainEpoch();
    private:



        Settings settings_;
        std::vector<TrainingInstanceType * > trainingSet_;

        // training edge contraction callbacks and graphs
        std::vector<TrainingInstanceDataType *> trainingSetData_;

        vigra::RandomForest<uint8_t> * classifier_;

        vigra::MultiArray<2, T> rfFeatures_;     
        vigra::MultiArray<2, uint8_t> rfLabels_;
    };



    template<class GRAPH, class T>
    void Gala<GRAPH,T>::train(){
       this->trainInitalRf();    
       this->trainEpoch();     
    }


    template<class GRAPH, class T>
    void Gala<GRAPH,T>::trainInitalRf(){
        std::cout<<"Start to train gala\n";

        NIFTY_CHECK(!trainingSet_.empty(), "training set must not be empty");


        // fetch some information from an arbitrary training example
        auto & someTrainingInstance = *trainingSet_.front();
        const uint64_t numberOfGraphs = trainingSet_.size();
        const uint64_t numberOfFeatures = someTrainingInstance.numberOfFeatures();

        


        

        // count the total number of initial edges
        // count the number of labeled edges
        auto initalNumberOfEdges = 0;
        auto initalNumberOfLabeledEdges = 0;
        for(size_t gi=0; gi<numberOfGraphs; ++gi){  
            initalNumberOfEdges += trainingSet_[gi]->initalNumberOfEdges();
            const auto & graph = trainingSet_[gi]->graph();
            const auto & edgeGt = trainingSet_[gi]->edgeGt();
            const auto & edgeGtUncertainty = trainingSet_[gi]->edgeGtUncertainty();
            for(const auto edge : graph.edges()){
                auto fgt = edgeGt[edge];
                auto ufgt = edgeGtUncertainty[edge];
                if((fgt < settings_.threshold0_ || fgt> settings_.threshold1_) && ufgt < settings_.thresholdU_){
                    ++initalNumberOfLabeledEdges;
                }
            }
        }

        std::cout<<"# Instances        : "<<trainingSet_.size()<<"\n";
        std::cout<<"# Features         : "<<numberOfFeatures<<"\n";
        std::cout<<"# InitalEdges      : "<<initalNumberOfEdges<<"\n";
        std::cout<<"# LabledInitalEdges: "<<initalNumberOfLabeledEdges<<"\n";


        // fill the initial training set
        
        // allocate the training data
        rfFeatures_.reshape(vigra::Shape2(initalNumberOfLabeledEdges,numberOfFeatures)); 
        rfLabels_.reshape(vigra::Shape2(initalNumberOfLabeledEdges,1));

        std::vector<T> fBuffer(numberOfFeatures);
        auto instanceIndex = 0;
        for(size_t gi=0; gi<numberOfGraphs; ++gi){  
            const auto & graph = trainingSet_[gi]->graph();
            auto features  = trainingSet_[gi]->features();
            const auto & edgeGt = trainingSet_[gi]->edgeGt();
            const auto & edgeGtUncertainty = trainingSet_[gi]->edgeGtUncertainty();
            for(const auto edge : graph.edges()){
                auto fgt = edgeGt[edge];
                auto ufgt = edgeGtUncertainty[edge];
                if((fgt < settings_.threshold0_ || fgt> settings_.threshold1_) && ufgt < settings_.thresholdU_){
                    const uint8_t label = fgt < settings_.threshold0_ ? 0 : 1;
                    features->getFeatures(edge, fBuffer.data());
                    for(auto fi=0; fi<numberOfFeatures; ++fi){
                        rfFeatures_(instanceIndex, fi) = fBuffer[fi];
                    }
                    rfLabels_(instanceIndex, 0) = label;
                    ++instanceIndex;
                }
            }
        }

        std::cout<<"learn classifier\n";
        if(classifier_ != nullptr){
            delete classifier_;
        }
        auto rfOpts = vigra::RandomForestOptions();  
        rfOpts.tree_count(100);
        rfOpts.predict_weighted();
        classifier_ = new  vigra::RandomForest<uint8_t>(rfOpts);

        // construct visitor to calculate out-of-bag error
        vigra::rf::visitors::OOB_Error oob_v;
        // perform training
        classifier_->learn(rfFeatures_, rfLabels_, vigra::rf::visitors::create_visitor(oob_v));
        std::cout << "the out-of-bag error is: " << oob_v.oob_breiman << "\n";

        
    }

    template<class GRAPH, class T>
    void Gala<GRAPH,T>::trainEpoch(){

        // do the initial prediction with the current random forest
        for(auto trainingInstanceData : trainingSetData_){  
            const auto & graph = trainingInstanceData->graph();
            for(const auto edge : graph.edges()){
                trainingInstanceData->recomputeFeaturesAndPredict(edge);
            }
        }


        
        for(auto trainingInstanceData : trainingSetData_){  



            const auto & pq = trainingInstanceData->pq_;
            auto & contractionGraph = trainingInstanceData->contractionGraph_;
        
            while(true){
                const auto nEdges = contractionGraph.numberOfEdges();
                const auto nNodes = contractionGraph.numberOfNodes();
                std::cout<<"#Edges "<<nEdges<<" #PQ "<<pq.size()<<" #Nodes "<<nNodes<<" #ufd "<< contractionGraph.ufd().numberOfSets()<<"\n";

                if(nEdges <=10){
                    break;
                }

                std::vector<uint64_t> edgesToContract;
                trainingInstanceData->toContract(edgesToContract);
                for(const auto edge : edgesToContract){
                    if(pq.contains(edge)){
                        contractionGraph.contractEdge(edge);
                    }   
                }
            }



        }

    }



} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_HXX
