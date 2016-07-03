#pragma once
#ifndef NIFTY_GRAPH_GALA_GALA_HXX
#define NIFTY_GRAPH_GALA_GALA_HXX

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
        typedef std::shared_ptr<FeatureBaseType>    FeatureBaseTypeSharedPtr;
        typedef Instance<GraphType, T>              InstanceType;
        typedef TrainingInstance<GraphType, T>      TrainingInstanceType;
        typedef GRAPH Graph;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;


        void addTrainingInstance(TrainingInstanceType * trainingInstance){
            trainingSet_.push_back(trainingInstance);
        }        
    private:
        std::vector<TrainingInstanceType * > trainingSet_;
    };


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_HXX
