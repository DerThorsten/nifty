#pragma once


#include <algorithm> // sort


#include "vigra/priority_queue.hxx"
#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty{
namespace graph{

// \cond SUPPRESS_DOXYGEN
namespace detail_watersheds_segmentation{




    template<
        class GRAPH,
        class NODE_WEIGHTS,
        class SEEDS,
        class LABELS
    >
    void nodeWeightedWatershedsSegmentationImpl(
        const GRAPH & g,
        const NODE_WEIGHTS      & nodeWeights,
        const SEEDS             & seeds,
        LABELS                  & labels
    ){  
        typedef GRAPH GraphType;
        typedef typename NODE_WEIGHTS::value_type WeightType;
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
                        const auto priority = nodeWeights[neigbour];
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
                        const auto priority = nodeWeights[targetNode];
                        pq.push(otherEdge,priority);
                       // inPQ[otherEdge]=true;
                    }
                }
            }
        }
    }






} // end namespace detail_watersheds_segmentation 

// \endcond


/// \brief edge weighted watersheds Segmentataion
/// 
/// \param g: input graph
/// \param nodeWeights : node weights / node height
/// \param seeds : seed must be non empty!
/// \param[out] labels : resulting  nodeLabeling (not necessarily dense)
template<class GRAPH,class NODE_WEIGHTS,class SEEDS,class LABELS>
void nodeWeightedWatershedsSegmentation(
    const GRAPH & g,
    const NODE_WEIGHTS & nodeWeights,
    const SEEDS        & seeds,
    LABELS             & labels
){  
    detail_watersheds_segmentation::nodeWeightedWatershedsSegmentationImpl(
        g,nodeWeights,seeds,labels);

}   
    





} // namespace nifty::graph
} // namespace nifty

