#pragma once
#ifndef NIFTY_GRAPH_GALA_GALA_INSTANCE_HXX
#define NIFTY_GRAPH_GALA_GALA_INSTANCE_HXX


namespace nifty{
namespace graph{

    
    template<class GRAPH, class T>
    class Instance{
    public:
        typedef GRAPH GraphType;
        typedef GalaFeatureBase<GraphType, T>     FeatureBaseType;

        Instance(
            const GraphType & graph, 
            FeatureBaseType * features
        )
        :   graph_(graph),
            features_(features){
        }
        const uint64_t numberOfFeatures(){
            return features_->numberOfFeatures();
        }
        const uint64_t initalNumberOfEdges(){
            return graph_.numberOfEdges();
        }
    protected:
        const GraphType & graph_;
        FeatureBaseType *  features_;
    };

    template<class GRAPH, class T>
    class TrainingInstance : public Instance<GRAPH, T> {
    public:
        typedef Instance<GRAPH,T> BaseType;
        typedef typename BaseType::GraphType GraphType;
        typedef typename BaseType::FeatureBaseType FeatureBaseType;
        typedef typename GraphType:: template EdgeMap<uint8_t>  EdgeGtType;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeGtType;

        template<class EDGE_GT>
        TrainingInstance(
            const GraphType & graph, 
            FeatureBaseType * features, 
            const EDGE_GT & edgeGt
        )
        :   BaseType(graph, features),
            edgeGt_(graph){
            for(const auto edge: graph.edges()){
                edgeGt_[edge] = edgeGt[edge];
            }
        }

        const uint64_t initalNumberOfLabeldEdges(){
            //return this->graph_.numberOfEdges();
        }

    private:
        EdgeGtType edgeGt_;
    };
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_INSTANCE_HXX
