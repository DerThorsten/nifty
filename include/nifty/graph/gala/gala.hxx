#pragma once
#ifndef NIFTY_GRAPH_GALA_GALA_HXX
#define NIFTY_GRAPH_GALA_GALA_HXX

#include <iostream>
#include "nifty/graph/gala/gala_feature_base.hxx"
#include "nifty/graph/gala/gala_instance.hxx"

namespace nifty{
namespace graph{

    


    template<class GRAPH, class T>
    class Gala{
    
    public:

        struct Settings{

        };

        typedef GRAPH GraphType;
        typedef GalaFeatureBase<GraphType, T>       FeatureBaseType;
        typedef Instance<GraphType, T>              InstanceType;
        typedef TrainingInstance<GraphType, T>      TrainingInstanceType;
        typedef GRAPH Graph;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;


        void addTrainingInstance(TrainingInstanceType * trainingInstance){
            trainingSet_.push_back(trainingInstance);
        }        

        void train();
    private:

        std::vector<TrainingInstanceType * > trainingSet_;
    };




    template<class GRAPH, class T>
    void Gala<GRAPH,T>::train(){
        std::cout<<"Start to train gala\n";

        NIFTY_CHECK(!trainingSet_.empty(), "training set must not be empty");


        // fetch some information from an arbitrary training example
        auto & someTrainingInstance = *trainingSet_.front();

        const uint64_t numberOfGraphs = trainingSet_.size();
        const uint64_t numberOfFeatures = someTrainingInstance.numberOfFeatures();

        

        // count the total number of initial edges
        auto initalNumberOfEdges = 0;
        for(size_t gi=0; gi<numberOfGraphs; ++gi){
            initalNumberOfEdges += trainingSet_[gi]->initalNumberOfEdges();
        }

        std::cout<<"# Instances  : "<<trainingSet_.size()<<"\n";
        std::cout<<"# Features   : "<<numberOfFeatures<<"\n";
        std::cout<<"# InitalEdges: "<<initalNumberOfEdges<<"\n";

        // allocate space for the inital training set
    }




} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_HXX
