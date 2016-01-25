#pragma once
#ifndef NIFTY_GRAPH_SHORTEST_PATH_HXX
#define NIFTY_GRAPH_SHORTEST_PATH_HXX

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/queue/changeable_priority_queue.hxx"

namespace nifty{
namespace graph{


    


    template<class GRAPH, class WEIGHT_TYPE>
    class ShortestPathDijkstra{

    public:
        typedef GRAPH Graph;
        typedef WEIGHT_TYPE WeightType;

        typedef typename Graph:: template NodeMap<int64_t>     PredecessorsMap;
        typedef typename Graph:: template NodeMap<WeightType>  DistanceMap;
    private:
        typedef nifty::queue::ChangeablePriorityQueue<WeightType>    PqType;
    public:
        ShortestPathDijkstra(const Graph & g)
        :   g_(g),
            pq_(g.maxNodeId()+1),
            predMap_(g),
            distMap_(g){
        }

        template<class EDGE_WEGIHTS, class SOURCE_T>
        void run(
            EDGE_WEGIHTS edgeWeights,
            const std::initializer_list<SOURCE_T> & sources
        ){
            this->run(edgeWeights, sources.begin(), sources.end());
        }


        template<class EDGE_WEGIHTS, class SOURCE_ITER>
        void run(
            EDGE_WEGIHTS edgeWeights,
            SOURCE_ITER sourceBegin, 
            SOURCE_ITER sourceEnd
        ){
            this->initializeMaps(sourceBegin, sourceEnd);

            // subgraph mask
            DefaultSubgraphMask<Graph> subgraphMask;

            // visitor
            auto visitor = []
            (   
                int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                return true;
            };
            this->runImpl(edgeWeights, sourceBegin, sourceEnd,subgraphMask,visitor);
        }

        const DistanceMap & distances()const{
            return distMap_;
        }
        const PredecessorsMap predecessors()const{
            return predMap_;
        }
    private:

        template<
            class EDGE_WEGIHTS, 
            class SOURCE_ITER,
            class SUBGRAPH_MASK,
            class VISITOR 
        >
        void runImpl(
            EDGE_WEGIHTS edgeWeights,
            SOURCE_ITER sourceBegin, 
            SOURCE_ITER sourceEnd,
            const SUBGRAPH_MASK &  subgraphMask,
            VISITOR && visitor
        ){

            //target_ = lemon::INVALID;
            while(!pq_.empty() ){ //&& !finished){
                const auto topNode =  pq_.top();
                visitor(topNode, distMap_, predMap_);
                pq_.pop();                
                if(subgraphMask.useNode(topNode)){
                    // loop over all neigbours
                    for(auto adj : g_.adjacency(topNode)){
                        auto otherNode = adj.node();
                        const auto edge = adj.edge();
                        if(subgraphMask.useNode(otherNode) && subgraphMask.useEdge(otherNode)){
                            if(pq_.contains(otherNode)){
                                const WeightType currentDist     = distMap_[otherNode];
                                const WeightType alternativeDist = distMap_[topNode]+edgeWeights[edge];
                                if(alternativeDist<currentDist){
                                    pq_.push(otherNode,alternativeDist);
                                    distMap_[otherNode]=alternativeDist;
                                    predMap_[otherNode]=topNode;
                                }
                            }
                            else if(predMap_[otherNode]==-1){
                                const WeightType initialDist = distMap_[topNode]+edgeWeights[edge];
                                //if(initialDist<=maxDistance)
                                //{
                                pq_.push(otherNode,initialDist);
                                distMap_[otherNode]=initialDist;
                                predMap_[otherNode]=topNode;
                                //}
                            }
                        }
                    }
                }
            }
            while(!pq_.empty() ){
                const auto topNode = pq_.top();
                predMap_[topNode] = -1;
                pq_.pop();
            }
        }


        template<class SOURCE_ITER>
        void initializeMaps(SOURCE_ITER sourceBegin, SOURCE_ITER sourceEnd){
                
            for(auto node : g_.nodes()){
                predMap_[node] = -1;
            }

            for( ; sourceBegin!=sourceEnd; ++sourceBegin){
                auto n = *sourceBegin;
                distMap_[n] = static_cast<WeightType>(0);
                predMap_[n] = n;
                pq_.push(n,static_cast<WeightType>(0));
            }
        }



        const GRAPH & g_;
        PqType pq_;
        PredecessorsMap predMap_;
        DistanceMap     distMap_;
    };



} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_SHORTEST_PATH_HXX
