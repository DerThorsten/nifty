#pragma once


#include <algorithm> // sort


#include "vigra/priority_queue.hxx"
#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty{
namespace graph{

// \cond SUPPRESS_DOXYGEN
namespace detail_watersheds_segmentation{

    struct RawPriorityFunctor{
        template<class LabelType, class T>
        T operator()(const LabelType /*label*/,const T  priority)const{
            return priority;
        }
    };


    template<class PRIORITY_TYPE,class LABEL_TYPE>
    struct CarvingFunctor{
        CarvingFunctor(const LABEL_TYPE backgroundLabel,
                       const PRIORITY_TYPE & factor,
                       const PRIORITY_TYPE & noPriorBelow
        )
        :   backgroundLabel_(backgroundLabel),
            factor_(factor),
            noPriorBelow_(noPriorBelow){
        }
        PRIORITY_TYPE operator()(const LABEL_TYPE label,const PRIORITY_TYPE  priority)const{
            if(priority>=noPriorBelow_)
                return (label==backgroundLabel_ ? priority*factor_ : priority);
            else{
                return priority;
            }
        }
        LABEL_TYPE     backgroundLabel_;
        PRIORITY_TYPE  factor_;
        PRIORITY_TYPE  noPriorBelow_;
    };



    template<
        class GRAPH,
        class EDGE_WEIGHTS,
        class SEEDS,
        class PRIORITY_MANIP_FUNCTOR,
        class LABELS
    >
    void edgeWeightedWatershedsSegmentationImpl(
        const GRAPH & g,
        const EDGE_WEIGHTS      & edgeWeights,
        const SEEDS             & seeds,
        PRIORITY_MANIP_FUNCTOR  & priorManipFunctor,
        LABELS                  & labels
    ){  
        typedef GRAPH GraphType;
        typedef typename EDGE_WEIGHTS::value_type WeightType;
        typedef typename LABELS::value_type  LabelType;
        //typedef typename Graph:: template EdgeMap<bool>    EdgeBoolMap;
        typedef vigra::PriorityQueue<int64_t, WeightType,true> PQ;

        PQ pq;
        for(auto node : g.nodes())
            labels[node] = seeds[node];
        
        // put edges from nodes with seed on pq
        for(auto node : g.nodes()){
            if(labels[node]!=static_cast<LabelType>(0)){

                for(auto adj : g.adjacency(node)){
                    const auto edge = adj.edge();
                    const auto neigbour = adj.node();
                    //std::cout<<"n- node "<<g.id(neigbour)<<"\n";
                    if(labels[neigbour]==static_cast<LabelType>(0)){
                        const auto priority = priorManipFunctor(labels[node],edgeWeights[edge]);
                        pq.push(edge,priority);
                        //inPQ[edge]=true;
                    }
                }
            }
        }


        while(!pq.empty()){

            const auto edge = pq.top();
            pq.pop();

            const auto u = g.u(edge);
            const auto v = g.v(edge);
            const LabelType lU = labels[u];
            const LabelType lV = labels[v];


            if(lU==0 && lV==0){
                throw std::runtime_error("both have no labels");
            }
            else if(lU!=0 && lV!=0){
                // nothing to do
            }
            else{

                const auto unlabeledNode = lU==0 ? u : v;
                const auto label = lU==0 ? lV : lU;

                // assign label to unlabeled node
                labels[unlabeledNode] = label;

                // iterate over the nodes edges
                for(auto adj : g.adjacency(unlabeledNode)){
                    const auto otherEdge = adj.edge();
                    const auto targetNode =  adj.node();
                    if(labels[targetNode] == 0){
                    //if(inPQ[otherEdge] == false && labels[targetNode] == 0){
                        const auto priority = priorManipFunctor(label,edgeWeights[otherEdge]);
                        pq.push(otherEdge,priority);
                       // inPQ[otherEdge]=true;
                    }
                }
            }
        }
    }


    template<
        class GRAPH,
        class EDGE_WEIGHTS,
        class SEEDS,
        class LABELS
    >
    void edgeWeightedWatershedsSegmentationKruskalImpl(
        const GRAPH & g,
        const EDGE_WEIGHTS      & edgeWeights,
        const SEEDS             & seeds,
        LABELS                  & labels
    ){
        typedef GRAPH GraphType;
        typedef typename EDGE_WEIGHTS::value_type WeightType;
        typedef typename LABELS::value_type  LabelType;
        //typedef typename Graph:: template EdgeMap<bool>    EdgeBoolMap;

        for(auto node : g.nodes()){
            labels[node] = seeds[node];
        }

        typedef std::pair<uint64_t, WeightType> IndexedWeight;
        std::vector< IndexedWeight > indexedWeights(g.numberOfEdges());

        auto c=0;
        for(auto edge : g.edges()){
            indexedWeights[c] = IndexedWeight(edge, edgeWeights[edge]);
            ++c;
        }

        // sort ascennding
        std::sort(indexedWeights.begin(), indexedWeights.end(),
            [](const IndexedWeight & a, const IndexedWeight & b) {
                return a.second < b.second;
            }
        );

        // merge
        nifty::ufd::Ufd<> ufd(g.nodeIdUpperBound()+1);

        for(const auto & indexWeight : indexedWeights){
            const auto edge = indexWeight.first;

            const auto uv = g.uv(edge);
            const auto u = uv.first;
            const auto v = uv.second;
            const auto ru = ufd.find(u);
            const auto rv = ufd.find(v);

            // not yet merged
            if(ru != rv){

                const auto lu = labels[ru];
                const auto lv = labels[rv];

                if( lu==0 || lv==0){

                    auto newLabel = std::max(lu, lv);
                    ufd.merge(u,v);
                    labels[u] = newLabel;
                    labels[v] = newLabel;
                    labels[ru] = newLabel;
                    labels[rv] = newLabel;
                }
            }
        }

        for(auto node : g.nodes()){
            labels[node] = labels[ufd.find(node)];
        }
    }



} // end namespace detail_watersheds_segmentation

// \endcond

    /// \brief edge weighted watersheds Segmentataion
    ///
    /// \param g: input graph
    /// \param edgeWeights : edge weights / edge indicator
    /// \param seeds : seed must be non empty!
    /// \param[out] labels : resulting  nodeLabeling (not necessarily dense)
    template<class GRAPH,class EDGE_WEIGHTS,class SEEDS,class LABELS>
    void edgeWeightedWatershedsSegmentation(
        const GRAPH & g,
        const EDGE_WEIGHTS & edgeWeights,
        const SEEDS        & seeds,
        LABELS             & labels
    ){
        detail_watersheds_segmentation::edgeWeightedWatershedsSegmentationKruskalImpl(
            g,edgeWeights,seeds,labels);
        //detail_watersheds_segmentation::RawPriorityFunctor fPriority;
        //detail_watersheds_segmentation::edgeWeightedWatershedsSegmentationImpl(g,edgeWeights,seeds,fPriority,labels);
    }


    /// \brief edge weighted watersheds Segmentataion
    ///
    /// \param g: input graph
    /// \param edgeWeights : edge weights / edge indicator
    /// \param seeds : seed must be non empty!
    /// \param backgroundLabel : which label is background
    /// \param backgroundBias  : bias for background
    /// \param noPriorBelow  : don't bias the background if edge indicator is below this value
    /// \param[out] labels : resulting  nodeLabeling (not necessarily dense)
    template<class GRAPH,class EDGE_WEIGHTS,class SEEDS,class LABELS>
    void carvingSegmentation(
        const GRAPH                              & g,
        const EDGE_WEIGHTS                       & edgeWeights,
        const SEEDS                              & seeds,
        const typename LABELS::value_type        backgroundLabel,
        const typename EDGE_WEIGHTS::value_type  backgroundBias,
        const typename EDGE_WEIGHTS::value_type  noPriorBelow,
        LABELS                      & labels
    ){
        typedef typename EDGE_WEIGHTS::Value WeightType;
        typedef typename LABELS::Value       LabelType;
        detail_watersheds_segmentation::CarvingFunctor<WeightType,LabelType> fPriority(backgroundLabel,backgroundBias, noPriorBelow);
        detail_watersheds_segmentation::edgeWeightedWatershedsSegmentationImpl(g,edgeWeights,seeds,fPriority,labels);
    }




} // namespace nifty::graph
} // namespace nifty

