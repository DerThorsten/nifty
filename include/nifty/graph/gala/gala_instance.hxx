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
        typedef std::shared_ptr<FeatureBaseType>  FeatureBaseTypeSharedPtr;

        Instance(
            const GraphType & graph, 
            FeatureBaseTypeSharedPtr features
        )
        :   graph_(graph),
            features_(features){
        }
    private:
        const GraphType & graph_;
        FeatureBaseTypeSharedPtr  features_;
    };

    template<class GRAPH, class T>
    class TrainingInstance : public Instance<GRAPH, T> {
    public:
        typedef Instance<GRAPH,T> BaseType;
        typedef typename BaseType::GraphType GraphType;
        typedef typename BaseType::FeatureBaseType FeatureBaseType;
        typedef typename BaseType::FeatureBaseTypeSharedPtr FeatureBaseTypeSharedPtr;
        typedef typename GraphType:: template EdgeMap<uint8_t>  EdgeGtType;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeGtType;

        template<class EDGE_GT>
        TrainingInstance(
            const GraphType & graph, 
            FeatureBaseTypeSharedPtr features, 
            const EDGE_GT & edgeGt
        )
        :   BaseType(graph, features),
            edgeGt_(graph){
            for(const auto edge: graph.edges()){
                edgeGt_[edge] = edgeGt[edge];
            }
        }
    private:
        EdgeGtType edgeGt_;
    };
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_INSTANCE_HXX
