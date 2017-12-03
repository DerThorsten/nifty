#pragma once

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/tools/changable_priority_queue.hxx"

namespace nifty{
namespace graph{

    template<class GRAPH, class WEIGHT_TYPE>
    class ShortestPathDijkstra{

    public:
        typedef GRAPH GraphType;
        typedef WEIGHT_TYPE WeightType;

        typedef typename GraphType:: template NodeMap<int64_t>     PredecessorsMap;
        typedef typename GraphType:: template NodeMap<WeightType>  DistanceMap;
    private:
        typedef nifty::tools::ChangeablePriorityQueue<WeightType>    PqType;
    public:
        ShortestPathDijkstra(const GraphType & g)
        :   g_(g),
            pq_(g.nodeIdUpperBound()+1),
            predMap_(g),
            distMap_(g){
        }

        // run single source single target
        // no  callback no mask exposed
        template<class EDGE_WEIGHTS>
        void runSingleSourceSingleTarget(
            const EDGE_WEIGHTS & edgeWeights,
            const int64_t source,
            const int64_t target = -1
        ){
            // subgraph mask
            DefaultSubgraphMask<GraphType> subgraphMask;
            // visitor
            auto visitor = [&]
            (
                int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                return topNode != target;
            };

            this->initializeMaps(&source, &source +1);
            runImpl(edgeWeights, subgraphMask, visitor);
        }

        // run single source multiple targets
        // no  callback no mask exposed
        template<class EDGE_WEGIHTS>
        void runSingleSourceMultiTarget(
            const EDGE_WEGIHTS & edgeWeights,
            const int64_t source,
            const std::vector<int64_t> & targets
        ){
            // subgraph mask
            DefaultSubgraphMask<GraphType> subgraphMask;
            // visitor
            size_t trgtsFound = 0;
            auto visitor = [&targets, &trgtsFound]
            (
                int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                if( std::find(targets.begin(), targets.end(), topNode) != targets.end() )
                    ++trgtsFound;
                if( trgtsFound >= targets.size() ) {
                    trgtsFound = 0;
                    return false;
                }
                return true;
            };

            this->initializeMaps(&source, &source +1);
            runImpl(edgeWeights, subgraphMask, visitor);
        }

        // run single source  ALL targets
        // no  callback no mask exposed
        template<class EDGE_WEIGHTS>
        void runSingleSource(
            const EDGE_WEIGHTS & edgeWeights,
            const int64_t source
        ){

            // subgraph mask
            DefaultSubgraphMask<GraphType> subgraphMask;
            this->initializeMaps(&source, &source +1);
            // visitor
            auto visitor = [](   int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                return true;
            };
            runImpl(edgeWeights, subgraphMask, visitor);
        }

        template<class EDGE_WEIGHTS, class SOURCE_ITER, class SUBGRAPH_MASK, class VISITOR>
        void run(
            const EDGE_WEIGHTS & edgeWeights,
            SOURCE_ITER sourceBegin,
            SOURCE_ITER sourceEnd,
            const SUBGRAPH_MASK &  subgraphMask,
            VISITOR && visitor
        ){
            this->initializeMaps(sourceBegin, sourceEnd);
            this->runImpl(edgeWeights,subgraphMask,visitor);
        }

        const DistanceMap & distances()const{
            return distMap_;
        }

        const PredecessorsMap & predecessors()const{
            return predMap_;
        }

        const GraphType & graph() const {
            return g_;
        }
    private:

        template<
            class EDGE_WEIGHTS,
            class SUBGRAPH_MASK,
            class VISITOR
        >
        void runImpl(
            const EDGE_WEIGHTS & edgeWeights,
            const SUBGRAPH_MASK & subgraphMask,
            VISITOR && visitor
        ){

            //target_ = lemon::INVALID;
            while(!pq_.empty() ){ //&& !finished){
                const auto topNode =  pq_.top();
                pq_.pop();

                if(!visitor(topNode, distMap_, predMap_)){
                    break;
                }
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

        const GraphType & g_;
        PqType pq_;
        PredecessorsMap predMap_;
        DistanceMap     distMap_;
    };

} // namespace nifty::graph
} // namespace nifty

