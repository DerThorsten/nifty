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
        const GraphType & graph() const{
            return graph_;
        }
        FeatureBaseType * features() {
            return features_;
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
        typedef typename GraphType:: template EdgeMap<double>  FuzzyEdgeGtType;
        typedef typename GraphType:: template EdgeMap<double>  EdgeGtUncertainty;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeGtType;

        template<class EDGE_GT>
        TrainingInstance(
            const GraphType & graph, 
            FeatureBaseType * features, 
            const EDGE_GT & edgeGt
        )
        :   BaseType(graph, features),
            edgeGt_(graph),
            edgeGtUncertainty_(graph,0.0){
            for(const auto edge: graph.edges()){
                edgeGt_[edge] = edgeGt[edge];
            }
        }

        template<class EDGE_GT, class EDGE_GT_UNCERTAINTY>
        TrainingInstance(
            const GraphType & graph, 
            FeatureBaseType * features, 
            const EDGE_GT & edgeGt__,
            const EDGE_GT_UNCERTAINTY & edgeGtUncertainty__
        )
        :   BaseType(graph, features),
            edgeGt_(graph),
            edgeGtUncertainty_(graph){
            for(const auto edge: graph.edges()){
                edgeGt_[edge] = edgeGt_[edge];
                edgeGtUncertainty_[edge] = edgeGtUncertainty_[edge];
            }
        }
        const FuzzyEdgeGtType & edgeGt()const{
            return edgeGt_;
        }
        const EdgeGtUncertainty & edgeGtUncertainty()const{
            return edgeGtUncertainty_;
        }
    private:
        FuzzyEdgeGtType edgeGt_;
        EdgeGtUncertainty edgeGtUncertainty_;
    };
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_GALA_INSTANCE_HXX
