#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LOSS_AUGMENTED_VIEW_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LOSS_AUGMENTED_VIEW_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX



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





    template<class WEIGHTED_MODEL>   
    class LossAugmentedViewLiftedMulticutObjective :  public
        LiftedMulticutObjectiveBase<
            LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>, 
            typename WEIGHTED_MODEL::GraphType, 
            typename WEIGHTED_MODEL::LiftedGraphType, 
            typename WEIGHTED_MODEL::WeightType
        >
    {   
    public:

        typedef WEIGHTED_MODEL WeightedModelType;
        typedef typename WeightedModelType::WeightsMapType NotAugmentedWeights;
        typedef typename WeightedModelType::GraphType       GraphType;
        typedef typename WeightedModelType::LiftedGraphType LiftedGraphType;
        typedef typename WeightedModelType::WeightType      WeightType;

        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabels;
        typedef typename GraphType:: template NodeMap<float>    NodeSizes;


        class WeightsMap{

        public:
            WeightsMap(
                const LiftedGraphType * liftedGraph,
                const NotAugmentedWeights * notAugmentedWeights,
                const NodeLabels * nodeGt,
                const NodeSizes *  nodeSizes
            )   
            :   liftedGraph_(liftedGraph),
                notAugmentedWeights_(notAugmentedWeights),
                nodeGt_(nodeGt),
                nodeSizes_(nodeSizes)
            {

            }

            WeightType operator[](const uint64_t edge)const{
                    
                const auto uv = liftedGraph_->uv(edge);

                const auto naw = notAugmentedWeights_->operator[](edge);

                // is cut in gt?
                const auto isCut = nodeGt_->operator[](uv.first) != nodeGt_->operator[](uv.second); 

                // value of the loss
                const auto l =  nodeSizes_->operator[](uv.first) * nodeSizes_->operator[](uv.second);

                // new weight encodes cost for edge beeing cut
                const auto w = isCut ? naw + l : naw - l;

                return w;   
            }
        private:
            const LiftedGraphType * liftedGraph_;
            const NotAugmentedWeights * notAugmentedWeights_;
            const NodeLabels * nodeGt_;
            const NodeSizes *  nodeSizes_;
        };


    private:
        //typedef nifty::graph::detail_graph::NodeIndicesToContiguousNodeIndices<GRAPH > ToContiguousNodes;


        typedef std::is_same<typename GraphType::NodeIdTag,  ContiguousTag> GraphHasContiguousNodeIds;

        static_assert( GraphHasContiguousNodeIds::value,
                  "LiftedMulticut assumes that the node id-s between graph and lifted graph are exchangeable \
                   The LossAugmentedViewLiftedMulticutObjective can only guarantee this for for graphs which have Contiguous Node ids "
        );

    public:

       

        template<class NODE_GT, class NODE_SIZES>
        LossAugmentedViewLiftedMulticutObjective(WeightedModelType & , const NODE_GT & , const NODE_SIZES & );


        WeightsMap & weights();
        const WeightsMap & weights() const;
        const GraphType & graph() const;
        const LiftedGraphType & liftedGraph() const;
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



        template<class WEIGHT_VECTOR>
        inline void changeWeights(const WEIGHT_VECTOR & weightVector);
        
        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void getGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;

        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void addGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;

        template<class NODE_LABELS, class GRADIENT_VECTOR>
        inline void substractGradient(const NODE_LABELS & ,GRADIENT_VECTOR &)const;


    protected:



        WeightedModelType & weightedModel_;

        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;

        NodeLabels nodeGt_;
        NodeSizes nodeSizes_;

        WeightsMap weightsMap_;

    };









    template<class WEIGHTED_MODEL>  
    template<class NODE_GT, class NODE_SIZES>
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    LossAugmentedViewLiftedMulticutObjective(
        WeightedModelType & weightedModel, 
        const NODE_GT & nodeGt, 
        const NODE_SIZES & nodeSizes
    )
    :   weightedModel_(weightedModel),
        graph_(weightedModel.graph()),
        liftedGraph_(weightedModel_.liftedGraph()),
        nodeGt_(weightedModel.graph()),
        nodeSizes_(weightedModel.graph()),
        weightsMap_( 
            &weightedModel.liftedGraph(),
            &weightedModel.weights(),
            &nodeGt_,
            &nodeSizes_
        )
    {
        graph_.forEachNode([&](const uint64_t node){
            nodeGt_[node] = nodeGt[node];
            nodeSizes_[node] = nodeSizes[node];
        });
    }



    template<class WEIGHTED_MODEL>    
    typename LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::WeightsMap & 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    weights(){
        return weightsMap_;
    }

    template<class WEIGHTED_MODEL>    
    const typename LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::WeightsMap & 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    weights() const{
        return weightsMap_;
    }

    template<class WEIGHTED_MODEL>    
    const typename LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::GraphType & 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    graph() const{
        return graph_;
    }

    template<class WEIGHTED_MODEL>    
    const typename LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::LiftedGraphType & 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    liftedGraph() const{
        return liftedGraph_;
    }
    

    template<class WEIGHTED_MODEL>    
    int64_t 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    graphEdgeInLiftedGraph(
        const uint64_t graphEdge
    )const{
        weightedModel_.graphEdgeInLiftedGraph(graphEdge);
    }

    template<class WEIGHTED_MODEL>    
    int64_t 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    liftedGraphEdgeInGraph(
        const uint64_t liftedGraphEdge
    )const{

        weightedModel_.liftedGraphEdgeInGraph(liftedGraphEdge);
    }

    template<class WEIGHTED_MODEL>    
    template<class F>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    forEachGraphEdge(
        F && f
    )const{
        weightedModel_.forEachGraphEdge(f);
    }

    template<class WEIGHTED_MODEL>    
    template<class F>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    parallelForEachGraphEdge(
        parallel::ThreadPool & threadpool,
        F && f
    )const{
        weightedModel_.parallelForEachGraphEdge(threadpool, f);
    }

    template<class WEIGHTED_MODEL>    
    template<class F>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    forEachLiftedeEdge(
        F && f
    )const{
        weightedModel_.forEachLiftedeEdge( f);
    }

    template<class WEIGHTED_MODEL>    
    template<class F>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    parallelForEachLiftedeEdge(
        parallel::ThreadPool & threadpool,
        F && f
    )const{

        weightedModel_.parallelForEachLiftedeEdge(threadpool, f);
    }




    template<class WEIGHTED_MODEL> 
    template<class WEIGHT_VECTOR>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    changeWeights(
        const WEIGHT_VECTOR & weightVector
    ){
        weightedModel_.changeWeights(weightVector);
    }



        
    template<class WEIGHTED_MODEL> 
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    getGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        weightedModel_.getGradient(gradient);
    }

    template<class WEIGHTED_MODEL> 
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    addGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        weightedModel_.addGradient(gradient);
    }

    template<class WEIGHTED_MODEL> 
    template<class NODE_LABELS, class GRADIENT_VECTOR>
    void 
    LossAugmentedViewLiftedMulticutObjective<WEIGHTED_MODEL>::
    substractGradient(const NODE_LABELS  & nodeLabels, GRADIENT_VECTOR & gradient) const {
        weightedModel_.substractGradient(gradient);
    }


  

} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LOSS_AUGMENTED_VIEW_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_HXX
