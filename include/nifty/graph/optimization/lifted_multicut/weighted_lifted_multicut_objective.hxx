#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_WEIGHTED_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_WEIGHTED_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX



#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_objective_base.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/parallel/threadpool.hxx"

#include "nifty/structured_learning/weight_vector.hxx"
#include "nifty/structured_learning/instances/weighted_edge.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{





    template<class GRAPH, class WEIGHT_TYPE>   
    class WeightedLiftedMulticutObjective :  public
        LiftedMulticutObjectiveBase<
            WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>, 
            GRAPH, UndirectedGraph<>, WEIGHT_TYPE
        >
    {   
    private:
        //typedef nifty::graph::detail_graph::NodeIndicesToContiguousNodeIndices<GRAPH > ToContiguousNodes;


        typedef std::is_same<typename GRAPH::NodeIdTag,  ContiguousTag> GraphHasContiguousNodeIds;

        static_assert( GraphHasContiguousNodeIds::value,
                  "LiftedMulticut assumes that the node id-s between graph and lifted graph are exchangeable \
                   The WeightedLiftedMulticutObjective can only guarantee this for for graphs which have Contiguous Node ids "
        );

    public:
        typedef GRAPH GraphType;


       


       

        typedef UndirectedGraph<> LiftedGraphType;

        typedef GraphType Graph;
        typedef LiftedGraphType LiftedGraph;
        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<LiftedGraph, WeightType> WeightsMapType;
        typedef WeightsMapType WeightsMap;
        









        WeightedLiftedMulticutObjective(const Graph & graph, const int64_t reserveAdditionalEdges = -1);

        std::pair<bool,uint64_t> setCost(const uint64_t u, const uint64_t v, const WeightType & w = 0.0, const bool overwrite = false);
        WeightsMap & weights();
        const WeightsMap & weights() const;
        const Graph & graph() const;
        const LiftedGraph & liftedGraph() const;
        int64_t graphEdgeInLiftedGraph(const uint64_t graphEdge)const;

        int64_t liftedGraphEdgeInGraph(const uint64_t liftedGraphEdge)const;

        /**
         * @brief Iterate over all edges of the lifted graph which are in the original graph
         * @details Iterate over all edges of the lifted graph which are in the original graph.
         * The ids are w.r.t. the lifted graph
         * 
         * @param f functor/lambda which is called for each edge id 
         */
        template<class F>
        void forEachGraphEdge(F && f)const;


        template<class F>
        void parallelForEachGraphEdge(parallel::ThreadPool &, F && f)const;


        /**
         * @brief Iterate over all edges of the lifted graph which are NOT in the original graph.
         * @details Iterate over all edges of the lifted graph which are NOT the original graph.
         * The ids are w.r.t. the lifted graph
         * 
         * @param f functor/lambda which is called for each edge id 
         */
        template<class F>
        void forEachLiftedeEdge(F && f)const;
        

        template<class F>
        void parallelForEachLiftedeEdge(parallel::ThreadPool & threadpool, F && f)const;



        template<class WEIGHT_INDICES_ITER, class FEATURE_ITER>
        std::pair<bool,uint64_t> addWeightedFeatures(const uint64_t u, const uint64_t v, 
                                 WEIGHT_INDICES_ITER, WEIGHT_INDICES_ITER, 
                                 FEATURE_ITER, const WeightType constTerm=0.0, 
                                 const bool overwriteConstTerm = false);



        std::pair<bool,uint64_t> addWeightedFeature(const uint64_t u, const uint64_t v, 
                                                    const uint64_t weightIndex, const WeightType feature);


        std::pair<bool,uint64_t> setConstTerm(const uint64_t u, const uint64_t v, const WeightType constTerm);
        std::pair<bool,uint64_t> addConstTerm(const uint64_t u, const uint64_t v, const WeightType constTerm);

        template<class WEIGHT_VECTOR>
        inline void changeWeights(const WEIGHT_VECTOR & weightVector);
        
        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void getGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;

        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void addGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;

        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void substractGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;



    protected:



        template<class NODE_LABELS, class GRADIENT_VECTOR, class BINARY_OPERATOR>
        inline void accumulateGradient(const NODE_LABELS & ,GRADIENT_VECTOR & ,BINARY_OPERATOR && )const;


        template<class F>
        std::pair<bool,uint64_t>  ensureEdge(const uint64_t u, const uint64_t v, F && f);

        typedef structured_learning::instances::WeightedEdge<WEIGHT_TYPE> WeightedEdgeType;
        typedef typename GraphType:: template EdgeMap<WeightedEdgeType> WeightedEdgeCosts;


        const Graph & graph_;
        LiftedGraph liftedGraph_;
        WeightsMap weights_;
        WeightedEdgeCosts weightedEdgeCosts_;

    };


















    template<class GRAPH, class WEIGHT_TYPE> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    WeightedLiftedMulticutObjective(
        const Graph & graph, 
        const int64_t reserveAdditionalEdges
    )
    :   graph_(graph),
        liftedGraph_(graph.numberOfNodes(), graph.numberOfEdges() + (reserveAdditionalEdges<0 ?  graph.numberOfEdges() : reserveAdditionalEdges) ), 
        weights_(liftedGraph_),
        weightedEdgeCosts_(liftedGraph_){

        for(const auto edge : graph_.edges()){
            const auto uv = graph_.uv(edge);
            liftedGraph_.insertEdge(
                uv.first,
                uv.second
            );
        }
        NIFTY_CHECK_OP(liftedGraph_.numberOfEdges(), == , graph_.numberOfEdges(),"");
        weights_.insertedEdges(liftedGraph_.edgeIdUpperBound(),0);
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    setCost(
        const uint64_t u, 
        const uint64_t v, 
        const WeightType & w , 
        const bool overwrite
    ){
        const auto preSize = liftedGraph_.numberOfEdges();
        const auto edge = liftedGraph_.insertEdge(u,v);
        if( liftedGraph_.numberOfEdges() > preSize){
            weights_.insertedEdges(edge, w);
            return std::pair<bool,uint64_t>(edge,true);
        }
        else{
            if(overwrite)
                weights_[edge] = w;
            else
                weights_[edge] += w;
            return std::pair<bool,uint64_t>(edge,false);
        }
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    typename WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::WeightsMap & 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    weights(){
        return weights_;
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    const typename WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::WeightsMap & 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    weights() const{
        return weights_;
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    const typename WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::Graph & 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    graph() const{
        return graph_;
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    const typename WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::LiftedGraph & 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    liftedGraph() const{
        return liftedGraph_;
    }
    

    template<class GRAPH, class WEIGHT_TYPE>   
    int64_t 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    graphEdgeInLiftedGraph(
        const uint64_t graphEdge
    )const{

        typedef std::is_same<typename Graph::EdgeIdTag,  ContiguousTag> CondA;
        typedef std::is_same<typename Graph::EdgeIdOrderTag, SortedTag> CondB;

        if(CondA::value && CondB::value  ){
            return graphEdge;
        }
        else{
            // this is not efficient, we should refactor this
            const auto uv = graph_.uv(graphEdge);
            return liftedGraph_.findEdge(uv.first, uv.second);
        }
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    int64_t 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    liftedGraphEdgeInGraph(
        const uint64_t liftedGraphEdge
    )const{

        typedef std::is_same<typename Graph::EdgeIdTag,  ContiguousTag> CondA;
        typedef std::is_same<typename Graph::EdgeIdOrderTag, SortedTag> CondB;

        if(CondA::value && CondB::value  ){
            if(liftedGraphEdge < graph_.numberOfEdges())
                return liftedGraphEdge;
            else
                return -1;
        }
        else{
            // this is not efficient, we should refactor this
            const auto uv = liftedGraph_.uv(liftedGraphEdge);
            return graph_.findEdge(uv.first, uv.second);
        }
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    template<class F>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    forEachGraphEdge(
        F && f
    )const{
        for(uint64_t e = 0 ; e<graph_.numberOfEdges(); ++e){
            f(e);
        }
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    template<class F>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    parallelForEachGraphEdge(
        parallel::ThreadPool & threadpool,
        F && f
    )const{
        parallel::parallel_foreach(threadpool,graph_.numberOfEdges(),
        [&](const int tid, const uint64_t e){
            f(tid, e);
        });
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    template<class F>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    forEachLiftedeEdge(
        F && f
    )const{
        for(uint64_t e = graph_.numberOfEdges(); e<liftedGraph_.numberOfEdges(); ++e){
            f(e);
        }
    }

    template<class GRAPH, class WEIGHT_TYPE>   
    template<class F>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    parallelForEachLiftedeEdge(
        parallel::ThreadPool & threadpool,
        F && f
    )const{

        const auto gEdgeNum =  graph_.numberOfEdges();
        parallel::parallel_foreach(threadpool,this->numberOfLiftedEdges(),
        [&](const int tid, const uint64_t i){
            const uint64_t e = i + gEdgeNum;
            f(tid, e);
        });
    }






    template<class GRAPH, class WEIGHT_TYPE>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addWeightedFeature(
        const uint64_t u, 
        const uint64_t v, 
        const uint64_t weightIndex, 
        const WeightType feature
    ){
        return this->ensureEdge(u, v,
        [&](WeightedEdgeType & weightedEdge){
            weightedEdge.addWeightedFeature(weightIndex, feature);
        });
    }


    template<class GRAPH, class WEIGHT_TYPE>
    template<class WEIGHT_INDICES_ITER, class FEATURE_ITER>
    std::pair<bool,uint64_t>  
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addWeightedFeatures(
        const uint64_t u, const uint64_t v,
        WEIGHT_INDICES_ITER weightIndicesBegin,  
        WEIGHT_INDICES_ITER weightIndicesEnd, 
        FEATURE_ITER featuresBegin,
        const WeightType constTerm, 
        const bool overwriteConstTerm
    ){
        
        return this->ensureEdge(u, v,[&](WeightedEdgeType & weightedEdge){
            while(weightIndicesBegin != weightIndicesEnd){
                weightedEdge.addWeightedFeature(*weightIndicesBegin, *featuresBegin);
                ++weightIndicesBegin;
                ++featuresBegin;
            }
            if(overwriteConstTerm)
                weightedEdge.setConstTerm(constTerm);
            else
                weightedEdge.addConstTerm(constTerm);
        });
    }

    template<class GRAPH, class WEIGHT_TYPE>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    setConstTerm(
        const uint64_t u, 
        const uint64_t v, 
        const WeightType constTerm
    ){
        return this->ensureEdge(u, v,[&](WeightedEdgeType & weightedEdge){
            weightedEdge.setConstTerm(constTerm);
        }); 
    }

    template<class GRAPH, class WEIGHT_TYPE>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addConstTerm(
        const uint64_t u, 
        const uint64_t v, 
        const WeightType constTerm
    ){
        return this->ensureEdge(u, v,[&](WeightedEdgeType & weightedEdge){
            weightedEdge.addConstTerm(constTerm);
        }); 
    }



    template<class GRAPH, class WEIGHT_TYPE>
    template<class F>
    std::pair<bool,uint64_t> 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    ensureEdge(
        const uint64_t u, 
        const uint64_t v, 
        F && f
    ){
        
        const auto ret = this->setCost(u, v);
        const uint64_t edge = ret.first;
        const bool addedNewEdge = ret.second;
        if(addedNewEdge){
            // new WeightedEdge
            WeightedEdgeType weightedEdge;
            weightedEdgeCosts_.insertedEdge(edge, weightedEdge);
            f(weightedEdgeCosts_[edge]);
        }
        else{
            // existing edge
            auto & weightedEdge = weightedEdgeCosts_[edge];
            f(weightedEdge);

        }
        return ret;
    }




    template<class GRAPH, class WEIGHT_TYPE>
    template<class WEIGHT_VECTOR>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    changeWeights(
        const WEIGHT_VECTOR & weightVector
    ){
        this->liftedGraph().forEachEdge([&](const uint64_t edge){
            this->weights_[edge] = weightedEdgeCosts_[edge].value(weightVector);
        });
    }



        
    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    getGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        for(auto i=0; i<gradient.size(); ++i){
            gradient[0] = 0.0;
        }
        this->accumulateGradient(nodeLabels, gradient, 
            [](const float a, const float b){
                return a+b;
            }
        );
    }

    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    addGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        this->accumulateGradient(nodeLabels, gradient, 
            [](const float a, const float b){
                return a+b;
            }
        );
    }

    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    substractGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {

    }


    template<class GRAPH, class WEIGHT_TYPE>
    template<class NODE_LABELS, class GRADIENT_VECTOR, class BINARY_OPERATOR>
    void 
    WeightedLiftedMulticutObjective<GRAPH, WEIGHT_TYPE>::
    accumulateGradient(
        const NODE_LABELS & nodeLabels,
        GRADIENT_VECTOR & gradient,
        BINARY_OPERATOR && binaryOperator
    )const{
        this->liftedGraph().forEachEdge([&](const uint64_t edge){
            const auto uv = this->uv(edge);
            if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                weightedEdgeCosts_[edge].accumulateGradient(gradient, binaryOperator);
            }
        });
    }

} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_WEIGHTED_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
